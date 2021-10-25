# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/8/27 16:11
@desc: compute recall and precision of knn-index
for each item (query, label) in test dstore, we query kNN and their corresponding labels,
then compute the recall count: how many knn labels is the same with label  #  precision and recall.
"""


import argparse
import logging

import faiss
import numpy as np
import torch
from tqdm import tqdm
import math

from knn.data_store import DataStore
from knn.path_utils import *
from knn.pq_wrapper import TorchPQCodec

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGING = logging.getLogger('knn.quantize-features')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="path to binary dataset directory")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    args = parser.parse_args()

    save_path = quantizer_path(args.data_dir)
    LOGGING.info(f"load pretrained quantizer at {save_path}")
    quantizer = faiss.read_index(save_path)
    quantizer = TorchPQCodec(quantizer)
    if args.use_gpu:
        LOGGING.info("Using gpu")
        quantizer = quantizer.cuda()

    qt_path = quantized_feature_path(args.data_dir, "train")
    qt_codes = np.load(qt_path)

    ds = DataStore.from_pretrained(dstore_dir=dstore_path(data_dir=args.data_dir, subset="train"))
    fp_codes = ds.keys
    total = fp_codes.shape[0]
    assert total == qt_codes.shape[0]
    chunk_size = 1024
    offset = 0
    total_error = 0
    pbar = tqdm(total=total)
    while offset < total:
        end = min(offset + chunk_size, total)
        bsz = end - offset
        fp_batch = torch.from_numpy(fp_codes[offset: end])  # [bsz, h]
        qt_batch = torch.from_numpy(qt_codes[offset: end])  # [bsz, c]
        if args.use_gpu:
            fp_batch = fp_batch.cuda()
            qt_batch = qt_batch.cuda()

        reconstruct_qt = quantizer.decode(qt_batch)
        avg_relative_error = (((fp_batch - reconstruct_qt) ** 2).sum(-1) / (fp_batch ** 2).sum(-1)).mean().item()
        total_error += avg_relative_error * bsz
        pbar.update(bsz)
        pbar.set_postfix({"L2 error": total_error/pbar.n})
        offset += bsz


if __name__ == '__main__':
    main()
