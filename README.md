# GNN-LM
## Introduction
This repo contains code for paper [GNN-LM: Language Modeling based on Global Contexts via GNN](https://arxiv.org/abs/2110.08743),
and is a fork of the [knnlm](https://github.com/urvashik/knnlm) repository from commit-id `fb6b50e48136b2c201f4768005474dc90e7791df`.

## Results
* Wiki103-Text

| Model | # Params | Test ppl |
|:------------|:-----------:|:-----------:|
| base LM | 247M | 18.7 | 
| + GNN | 274M | 16.8 |
| + GNN + KNN | 274M | 14.8 |

* One Billion Dataset

| Model | # Params | Test ppl |
|:------------|:-----------:|:-----------:|
| base LM | 247M | 18.7 | 
| + GNN | 274M | 16.8 |
| + GNN + KNN | 274M | 14.8 |

* EnWiki8

| Model | # Params | Test BPC |
|:------------|:-----------:|:-----------:|
| base LM | 41M | 1.06 | 
| + GNN | 48M | 1.04 |
| + GNN + KNN | 48M | 1.03 |

## Requirements
* Python >= 3.6
* [PyTorch](https://pytorch.org/) >= 1.7.1
* [faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) >= 1.5.3(`pip install faiss-gpu` works for me, but it is not officially released by faiss team.)
* `pip install -r requirements.txt`
* `pip install -e .`

### A Note about Hardware

Experiments for this paper were conducted on machines that contain 500GB of RAM, NVIDIA V100 32GB GPUs and flash storage (SSDs). Saving the Wikitext-103 datastore requires 400GB of disk space. The speed of saving the datastore, building the FAISS index and evaluating the nearest neighbors language model heavily depends on the amount of RAM available for each job. Some of these steps can be sped up by parallelizing, which we leave for users to do in order to best cater to their setup.

If you are working with a remote cluster, please note that we use [memmaps](https://numpy.org/doc/1.18/reference/generated/numpy.memmap.html) for saving the datastore. This allows us to keep the data on disk while accessing it by loading small chunks into memory, depending on the available RAM. This means there are a large number of disk seeks. In order to prevent slowing down your entire cluster, we suggest always reading/writing this data to/from local disks (as opposed to NFS directories), and flash storage is best for faster access.

### Preparing the Data & Pretrained Models
* WikiText103: see `gnnlm_scripts/wiki103/prepare_wiki103.sh`, which includes downloading/preprocessing WikiText-103 dataset, reproducing our base LM, evaluation, and feature extraction.
* One Billion Word: see `gnnlm_scripts/one_billion/prepare_1billion.sh`
* Enwik8: see `gnnlm_scripts/enwik8/prepare_enwik8.sh`

### KNN Search and Feature Quantization
* WikiText103: see `gnnlm_scripts/wiki103/find_knn.sh`

### Training/Evaluation GNN-LM
* WikiText103: See `gnnlm_scripts/wiki103/hgt_lm_wiki103_reproduce.sh`

## TODOs
- [ ] Scripts for Enwik8
- [ ] Scripts for One Billion Dataset