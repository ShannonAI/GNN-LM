# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/7/5 19:54
@desc: 

"""

import os


def feature_path(data_dir, mode):
    return os.path.join(data_dir, f"{mode}_dstore", "keys.npy")


def value_path(data_dir, mode):
    return os.path.join(data_dir, f"{mode}_dstore", "vals.npy")


def dstore_path(data_dir, subset):
    return os.path.join(data_dir, f"{subset}_dstore")


def quantized_feature_path(data_dir, mode):
    return os.path.join(data_dir, f"{mode}_dstore", "quantized-keys.npy")


def quantizer_path(data_dir, suffix="", norm=False):
    return os.path.join(data_dir, f"quantizer{'-norm' if norm else ''}{suffix}")


def neighbor_path(data_dir, mode, k=32):
    return os.path.join(data_dir, f"{mode}_dstore", f"neighbors.mmap.{k}")


def dictionary_path(data_dir):
    return os.path.join(data_dir, f"dict.txt")


def fairseq_dataset_path(data_dir, mode):
    return os.path.join(data_dir, mode)