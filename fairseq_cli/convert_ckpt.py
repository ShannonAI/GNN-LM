# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/4/23 19:26
@desc: transform a transformer ckpt to tgtransformer

"""

import os

import torch
from fairseq import checkpoint_utils
from knn.pq_wrapper import TorchPQCodec
import faiss


TRANSFORMER_CKPT = "/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline/checkpoint_best.pt"
# OUT_CKPT = "/userhome/yuxian/train_logs/lm/wiki-103/fairseq_baseline/checkpoint_best_quantize128.pt"
# QUANTIZER = "/userhome/yuxian/data/lm/wiki-103/data-bin/quantizer"
TRANSFORMER_CKPT = "/data/yuxian/wiki103-yunnao/baseline/checkpoint_best.pt"
OUT_CKPT = "/data/yuxian/wiki103-yunnao/baseline/checkpoint_best_qt128.pt"
QUANTIZER = "/data/yuxian/wiki103-yunnao/data-bin/quantizer"
# update gcn quantizer from 64 to 128
# TRANSFORMER_CKPT = "/userhome/yuxian/train_logs/lm/wiki-103/0712_gcn_bidirect_noleak_n256_k128_adam_lr2e-5/checkpoint_best.pt"
# OUT_CKPT = "/userhome/yuxian/train_logs/lm/wiki-103/0712_gcn_bidirect_noleak_n256_k128_adam_lr2e-5/checkpoint_best.pt.128"
#
# TRANSFORMER_CKPT = "/userhome/yuxian/train_logs/lm/enwik8/fairseq_transformer_xl/ckpt/model.pt.convert"
# OUT_CKPT = "/userhome/yuxian/train_logs/lm/enwik8/fairseq_transformer_xl/ckpt/model.pt.convert.newquant"
#
# TRANSFORMER_CKPT = "/data/yuxian/wiki103-yunnao/baseline/checkpoint_best.pt"
# OUT_CKPT = "/data/yuxian/wiki103-yunnao/baseline/checkpoint_best_quantize.pt"
# QUANTIZER= "/data/yuxian/wiki103-yunnao/data-bin/quantizer"

state = checkpoint_utils.load_checkpoint_to_cpu(TRANSFORMER_CKPT)

# laod quantizer
# QUANTIZER = TorchPQCodec(index=faiss.read_index("//data/yuxian/datasets/wikitext-103/data-bin/quantizer"))
# QUANTIZER = TorchPQCodec(index=faiss.read_index("/userhome/yuxian/data/lm/wiki-103/data-bin/quantizer"))
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
