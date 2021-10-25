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
from knn.knn_model import KNNModel
from knn.data_store import DataStore
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dstore", required=True)
    parser.add_argument("--train-dstore", required=True)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--index", required=True)
    parser.add_argument("--cuda", default=-1, type=int)
    parser.add_argument("--max", default=-1, type=int)
    parser.add_argument("--nprobe", default=32, type=int)
    args = parser.parse_args()

    model = KNNModel(
        args.index, args.train_dstore, cuda=args.cuda,
        metric_type="do_not_recomp_ip",
        k=args.k,
        probe=args.nprobe,
    )
    test_dstore = DataStore.from_pretrained(args.test_dstore)
    max_num = test_dstore.dstore_size
    if args.max > 0:
        max_num = min(max_num, args.max)
    pbar = tqdm(total=max_num)
    c = 0
    position_recall = np.zeros([100])
    # position_c = np.zeros([100])
    # position = 0
    for i in range(max_num):
        query = np.array(test_dstore.keys[i:i+1])  # [batch, d], batch = 1
        if "cosine" in args.index:  # cosine metrics need normalized first
            query = query / np.sqrt((query ** 2).sum(-1, keepdims=True))
        label = np.array(test_dstore.vals[i]).reshape((query.shape[0], 1))  # [batch, 1]
        dists, knns = model.get_knns(query, args.k)  # [batch, k]
        print(dists)
        knn_labels = model.vals[knns]  # [batch, k]
        recall = (knn_labels == label).sum()
        c += recall
        pbar.update(query.shape[0])
        pbar.set_postfix({"avg_c": f"{c/pbar.n:.1f}"})

        # position_recall[position] += recall
        # position_c[position] += 1
        # position += 1
        # if label[0][0] == 2:
        #     position = 0

    # print(position_recall / position_c)


if __name__ == '__main__':
    main()
