# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/7/6 14:29
@desc: 

"""

import argparse
from knn.path_utils import *
from knn.data_store import DataStore
from knn.knn_model import KNNModel
import os
from tqdm import tqdm
import numpy as np
import logging


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGING = logging.getLogger('knn.find_knn')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="data directory")
    parser.add_argument("--subset", type=str, required=True, help="train/valid/test, find knn of which subset")
    parser.add_argument("--candidate_subset", type=str, default="train", help="find knn from which subset")
    parser.add_argument("--cuda", default=-1, type=int, help="cuda device")
    parser.add_argument("--nprobe", type=int, default=32, help="nprobe for faiss approximate knn search")
    parser.add_argument("--efsearch", type=int, default=8, help="efsearch for faiss approximate knn search")
    parser.add_argument("--k", type=int, default=32, help="find k nearest neighbors")
    parser.add_argument("--bsz", type=int, default=1024, help="batch size")
    args = parser.parse_args()

    data_dir = args.data_dir
    subset = args.subset

    ds_dir = dstore_path(data_dir, subset)
    ds = DataStore.from_pretrained(ds_dir)
    keys = ds.keys
    knn_model = KNNModel(
        index_file=os.path.join(dstore_path(data_dir, args.candidate_subset), "faiss_store.cosine"),
        dstore_dir=ds_dir,
        no_load_keys=True, use_memory=True, cuda=args.cuda,
        probe=args.nprobe,
        efsearch=args.efsearch
    )
    dstore_size = ds.dstore_size

    neighbor_file = neighbor_path(data_dir, subset, args.k)
    neighbor_array = np.memmap(neighbor_file, mode="w+", shape=(dstore_size, args.k), dtype=np.int64)

    start = 0

    pbar = tqdm(total=dstore_size)
    while start < dstore_size:
        end = min(start+args.bsz, dstore_size)
        batch_query = keys[start: end].astype(np.float32)

        knn_dists, knns = knn_model.get_knns(queries=batch_query, k=args.k)
        neighbor_array[start: end] = knns

        batch_size = end - start
        pbar.update(batch_size)
        start = end
    print(f"Save neighbor of shape {neighbor_array.shape} to {neighbor_file}")


if __name__ == '__main__':
    main()
