# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/4/23 19:26
@desc: convert a transformer ckpt to gnn-transformer

"""

import argparse
import os

import faiss
import torch

from fairseq import checkpoint_utils
from knn.pq_wrapper import TorchPQCodec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="original pretrained ckpt")
    parser.add_argument("--out", required=True, help="new ckpt save path")
    parser.add_argument("--quantizer", required=True, help="quantizer generated by faiss")
    args = parser.parse_args()

    # original pretrained ckpt
    # TRANSFORMER_CKPT = "/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline/checkpoint_best.pt"
    TRANSFORMER_CKPT = args.ckpt
    # target save path
    # OUT_CKPT = "/data/yuxian/wiki103-yunnao/baseline/checkpoint_best_qt128.pt"
    OUT_CKPT = args.out
    # quantizer generated by faiss
    QUANTIZER = args.quantizer

    state = checkpoint_utils.load_checkpoint_to_cpu(TRANSFORMER_CKPT)

    # load quantizer
    QUANTIZER = TorchPQCodec(index=faiss.read_index(QUANTIZER))
    state["model"]["decoder.tgt_quantizer.centroids_torch"] = QUANTIZER.centroids_torch
    state["model"]["decoder.tgt_quantizer.norm2_centroids_torch"] = QUANTIZER.norm2_centroids_torch
    state["model"]["decoder.tgt_quantizer.sdc_table_torch"] = QUANTIZER.sdc_table_torch
    if QUANTIZER.pre_torch:
        state["model"]["decoder.tgt_quantizer.A"] = QUANTIZER.A
        state["model"]["decoder.tgt_quantizer.b"] = QUANTIZER.b

    state["args"].graph = True

    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save(state, OUT_CKPT)
    print(f"Saved ckpt to {OUT_CKPT}")


if __name__ == '__main__':
    main()
