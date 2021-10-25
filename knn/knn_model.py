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
from knn.data_store import DataStore

LOGGING = logging.getLogger(__name__)


class KNNModel(object):
    """通过查询FAISS的KNN计算LM log prob"""

    def __init__(self, index_file, dstore_dir, probe: int = 32, no_load_keys: bool = False,
                 metric_type: str = "do_not_recomp_ip", sim_func: str = None, k: int = 1024, cuda: int = -1,
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
        assert self.metric_type in ["do_not_recomp_l2", "do_not_recomp_ip", "l2", "ip"]
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
            except Exception as e:
                LOGGING.warning(f"index {self.index_file} does not support GPU", exc_info=True)
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
        dists, knns = self.index.search(queries.astype(np.float32), k=k)
        return dists, knns

    def get_knn_prob(
        self,
        queries,
        k: int = 0,
        output_size: int = None,
        return_knn: bool = False,
        t: float = 1.0,
        targets: torch.Tensor = None,
        return_recall: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, np.array, np.array], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            queries: Tensor of shape [batch, hidden]
            output_size: int
            k: int, number of neighbors
            return_knn: if True, return the knn dists and knn vals
            t: temperature
            targets: Tensor of shape [batch]

        Returns:
            if target is None, returns 3 values:
                probs: tensor of shape [batch, output_size]
                knn dists: np.array of shape [batch, K]
                knn keys: np.array of shape [batch, K]
            if target is not None, return probs of target: [batch]

        """
        device = queries.device

        assert self.data_store.val_size == 1, "make sure self.data_store.val_size == 1 (which is labels)"
        k = k or self.k
        if not (output_size or self.vocab_size):
            raise ValueError("DataStore.info does not have vocab_size，please set output_size manually")

        def sim_func(dists, knns, queries, function=None):
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
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            if function == 'do_not_recomp_l2':
                return -1 * dists

            if function == 'do_not_recomp_ip':
                return dists

            qsize = queries.size()

            if function == 'l2':
                # [batch, k, hidden]
                knns_vecs = torch.from_numpy(self.keys[knns].astype(np.float32)).to(device)
                # [batch, k, hidden]
                query_vecs = queries.view(qsize[0], 1, qsize[1]).repeat(1, k, 1)
                l2 = torch.sum((query_vecs - knns_vecs) ** 2, dim=2)
                return -1 * l2

            if function == 'ip':
                keys = torch.from_numpy(self.keys[knns].astype(np.float32)).to(device)  # [batch, k, hidden]

                if "cosine" in self.index_file:  # cosine metrics need normalized first
                    keys = keys / (keys ** 2).sum(-1, keepdims=True).sqrt()

                return (keys * queries.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            raise ValueError("Invalid knn similarity function!")

        # [batch, k]; [batch, k]

        if "cosine" in self.index_file:  # cosine metrics need normalized first
            knn_queries = queries / (queries ** 2).sum(-1, keepdims=True).sqrt()
        else:
            knn_queries = queries

        dists, knns = self.get_knns(knn_queries, k=k)
        dists = torch.from_numpy(dists).to(device)
        # [batch, k]
        sims = sim_func(dists, knns, knn_queries, function=self.metric_type)
        assert len(sims.size()) == 2

        # ignore -1(in faiss, -1 means padding item)
        sims.masked_fill_(torch.from_numpy(knns == -1).to(device), value=-1e10)

        # [batch, k]
        probs = F.softmax(sims / t, dim=-1)
        # [batch, k]
        knn_vals = torch.from_numpy(self.vals[knns]).long().to(device)
        batch_size = probs.shape[0]
        output_size = output_size or self.vocab_size

        if targets is None:
            weighted_probs = torch.zeros([batch_size, output_size], device=device)
            # weighted_probs[i][knn_vals[i][j]] += probs[i][j]
            weighted_probs.scatter_add_(dim=1, index=knn_vals, src=probs)
            if return_knn:
                return weighted_probs, sims, knns
            return weighted_probs

        probs = probs.to(device)
        target_probs = torch.zeros([batch_size, 2], device=device)
        target_mask = knn_vals == targets.unsqueeze(-1)  # [batch, k]
        target_probs.scatter_add_(dim=1, index=target_mask.long(), src=probs)

        if not return_recall:
            return target_probs[:, 1]
        return target_probs[:, 1], torch.sum(target_mask, dim=-1)


if __name__ == '__main__':
    dstore_dir = "/data/yuxian/datasets/wikitext-103/data-bin/train_dstore"
    # dstore_dir = "/userhome/yuxian/data/lm/wiki-103/data-bin/train_dstore"
    index_file = os.path.join(dstore_dir, "faiss_store.cosine")
    model = KNNModel(
        index_file, dstore_dir, cuda=-1,
        # metric_type="do_not_recomp_ip",
        metric_type="ip",
        # metric_type="l2",
        k=1024,
    )
    x = torch.from_numpy(model.data_store.keys[:32, :].astype(np.float32)).cuda()
    p = model.get_knn_prob(x)
    print(p.shape)
    p2 = model.get_knn_prob(x, targets=torch.from_numpy(model.data_store.vals[:32]), t=0.02)
    print(p2)
    print(p2.mean())
