# encoding: utf-8
"""



@version: 1.0
@file: knn_model
"""

import os
from time import time
from typing import Tuple, Union

import faiss
import numpy as np
import torch
from torch.nn import functional as F

import logging
from .data_store import DataStore

LOGGING = logging.getLogger(__name__)


class KNNModel(object):
    """通过查询FAISS的KNN计算LM log prob"""

    def __init__(self, index_file, dstore_dir, probe: int = 32, no_load_keys: bool = False,
                 metric_type: str = "do_not_recomp_l2", sim_func: str = None, k: int = 1024, cuda: int = -1,
                 use_memory=False, efsearch=8):

        self.index_file = index_file
        self.dstore_dir = dstore_dir
        self.probe = probe
        self.efsearch = efsearch
        self.no_load_keys = no_load_keys
        self.use_memory = use_memory
        t = time()
        self.data_store = DataStore.from_pretrained(dstore_dir=dstore_dir,
                                                    use_memory=use_memory,
                                                    no_load_keys=no_load_keys)
        LOGGING.info(f'Reading datastore took {time() - t} s')

        self.dstore_size = self.data_store.dstore_size
        self.hidden_size = self.data_store.hidden_size
        self.vocab_size = self.data_store.vocab_size
        self.dstore_fp16 = self.data_store.dstore_fp16
        self.vals = self.data_store.vals
        if not no_load_keys:
            self.keys = self.data_store.keys

        self.k = k
        self.metric_type = metric_type
        self.sim_func = sim_func
        self.index = self.setup_faiss()
        if cuda != -1:
            try:
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                # here we are using a 64-byte PQ, so we must set the lookup tables to
                # 16 bit float (this is due to the limited temporary memory).
                co.useFloat16 = True
                self.index = faiss.index_cpu_to_gpu(res, cuda, self.index, co)
                LOGGING.info(f"use gpu for index search")
            except Exception as e:
                LOGGING.error(f"index {self.index_file} does not support GPU", exc_info=1)
            cuda = -1
        self.use_memory = use_memory
        self.cuda = cuda

    def setup_faiss(self):
        """setup faiss index"""
        if not os.path.exists(self.dstore_dir):
            raise ValueError(f'Dstore path not found: {self.dstore_dir}')

        start = time()
        LOGGING.info(f'Reading faiss index, with nprobe={self.probe},  efSearch={self.efsearch} ...')
        index = faiss.read_index(self.index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
        LOGGING.info(f'Reading faiss of size {index.ntotal} index took {time() - start} s')
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", self.probe)
            faiss.ParameterSpace().set_index_parameter(index, "quantizer_efSearch", self.efsearch)
        except Exception as e:
            LOGGING.warning(f"faiss index {self.index_file} does not have parameter nprobe or efSearch")
        return index

    def get_knns(self, queries: Union[torch.Tensor, np.array], k: int = 0) -> Tuple[np.array, np.array]:
        """
        get distances and knns from queries
        Args:
            queries: Tensor of shape [num, hidden]
            k: number of k, default value is self.k
        Returns:
            dists: knn dists. np.array of shape [num, k]
            knns: knn ids. np.array of shape [num, k]
        """
        k = k or self.k
        if isinstance(queries, torch.Tensor):
            queries = queries.detach().cpu().float().data.numpy()
        dists, knns = self.index.search(queries, k=k)
        return dists, knns

    def get_knn_prob(
        self,
        queries,
        k: int = 0,
        output_size: int = None,
        return_knn: bool = False,
        t: float = 1.0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.array, np.array]]:
        """
        Args:
            queries: Tensor of shape [batch, hidden]
            output_size: int
            k: int, number of neighbors
            return_knn: if True, return the knn dists and knn vals
            t: temperature

        Returns:
            probs: tensor of shape [batch, output_size]
            knn dists: np.array of shape [batch, K]
            knn keys: np.array of shape [batch, K]

        """
        assert self.data_store.val_size == 1, "make sure self.data_store.val_size == 1 (which is labels)"
        k = k or self.k
        if not (output_size or self.vocab_size):
            raise ValueError("DataStore.info没有vocab_size，需要指定output_size")

        def dist_func(dists, knns, queries, function=None):
            """
            计算L2 distance
            Args:
                dists: knn distances, [batch, k]
                knns: knn ids, [batch, k]
                queries: qeuries, [batch, hidden]
                function: sim function
                k: number of neighbors

            Returns:
                dists. tensor of shape [batch, k]

            """
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = queries.size()
                if self.metric_type == 'l2':
                    # [batch, k, hidden]
                    knns_vecs = torch.from_numpy(self.keys[knns])  # .cuda().view(qsize[0], self.k, -1)
                    if self.dstore_fp16:
                        knns_vecs = knns_vecs.half()
                    # [batch, k, hidden]
                    query_vecs = queries.view(qsize[0], 1, qsize[1]).repeat(1, k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs) ** 2, dim=2)
                    return -1 * l2
                return dists

            if function == 'dot':
                qsize = queries.size()
                keys = torch.from_numpy(self.keys[knns])  # .cuda()
                return (keys * queries.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * dists

            raise ValueError("Invalid knn similarity function!")

        # [batch, k]; [batch, k]
        dists, knns = self.get_knns(queries, k=k)
        dists = torch.from_numpy(dists)  ##.cuda()
        # [batch, k]
        dists = dist_func(dists, knns, queries, function=self.sim_func)
        assert len(dists.size()) == 2
        # [batch, k]
        probs = F.softmax(dists / t)
        # [batch, k]
        knn_vals = torch.from_numpy(self.vals[knns]).long()  ##.cuda()
        batch_size = probs.shape[0]
        output_size = output_size or self.vocab_size
        weighted_probs = torch.zeros([batch_size, output_size])
        # weighted_probs[i][knn_vals[i][j]] += probs[i][j]
        weighted_probs.scatter_add_(dim=1, index=knn_vals, src=probs)
        if return_knn:
            return weighted_probs, dists, knns
        return weighted_probs
