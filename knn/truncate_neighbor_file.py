# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/6/12 15:26
@desc: truncate neighbor file smaller sizes
"""
from tqdm import tqdm
import numpy as np
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, type=str)
parser.add_argument("--src-k", required=True, type=int)
parser.add_argument("--tgt-k", required=True, type=int)
parser.add_argument("--subsets", nargs="+", default=["train", "valid", "test"])

args = parser.parse_args()


# data_dir = "/userhome/yuxian/data/lm/enwik8/data-bin"
data_dir = args.data
subsets = args.subsets
# subsets = ["test"]
num_tokens = []
for mode in subsets:
    info_file = os.path.join(data_dir, f"{mode}_dstore", "info.json")
    tokens = int(json.load(open(info_file))["dstore_size"])
    num_tokens.append(tokens)


bsz = 8192
origin_k = args.src_k
tgt_k = args.tgt_k


for subset, length in zip(subsets, num_tokens):
    src_file = f"{data_dir}/{subset}_dstore/neighbors.mmap.{origin_k}"
    src_shape = (length, origin_k)
    tgt_file = f"{data_dir}/{subset}_dstore/neighbors.mmap.{tgt_k}"
    tgt_shape = (length, tgt_k)
    assert src_file != tgt_file
    assert not os.path.exists(tgt_file)
    src_array = np.memmap(src_file, dtype=np.int64, shape=src_shape)
    tgt_array = np.memmap(tgt_file, dtype=np.int64, shape=tgt_shape, mode="w+")

    start = 0
    pbar = tqdm(total=length, desc="convert")
    while start < length:
        end = min(length, start + bsz)
        tgt_array[start: end] = src_array[start: end, : tgt_k]
        pbar.update(end - start)
        start = end
