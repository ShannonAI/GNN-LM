# GCN-LM

This repository is a fork of the [knnlm](https://github.com/urvashik/knnlm) repository and the exact commit id is `fb6b50e48136b2c201f4768005474dc90e7791df`.

## Requirements
* Python >= 3.6
* [PyTorch](https://pytorch.org/) >= 1.7.1
* [faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) >= 1.5.3(pip install faiss-gpu works for me, but it is not officially released by faiss team.)
* `pip install -r yuxian_requirements.txt`
* `pip install -e .`

### A Note about Hardware

Experiments for this paper were conducted on machines that contain 500GB of RAM, NVIDIA V100 32GB GPUs and flash storage (SSDs). Saving the Wikitext-103 datastore requires 400GB of disk space. The speed of saving the datastore, building the FAISS index and evaluating the nearest neighbors language model heavily depends on the amount of RAM available for each job. Some of these steps can be sped up by parallelizing, which we leave for users to do in order to best cater to their setup.

If you are working with a remote cluster, please note that we use [memmaps](https://numpy.org/doc/1.18/reference/generated/numpy.memmap.html) for saving the datastore. This allows us to keep the data on disk while accessing it by loading small chunks into memory, depending on the available RAM. This means there are a large number of disk seeks. In order to prevent slowing down your entire cluster, we suggest always reading/writing this data to/from local disks (as opposed to NFS directories), and flash storage is best for faster access.

### Preparing the Data
See `hgt_scripts/wiki103/prepare_wiki103.sh`, which includes downloading/preprocessing WikiText-103 dataset, reproducing our base LM, evaluation, and feature extraction.

### KNN Search and Feature Quantization

See `hgt_scripts/wiki103/find_knn.sh`

### Training GCN-LM
See `hgt_scripts/wiki103/hgt_lm_wiki103_reproduce.sh`
