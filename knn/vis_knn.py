# encoding: utf-8
"""



@desc: visualize knn neighbors

"""

import numpy as np
from fairseq.data import data_utils, MMapIndexedDataset
from fairseq.tasks.translation import TranslationTask
from knn.path_utils import *
from termcolor import colored
from knn.data_store import DataStore

import logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGING = logging.getLogger('knn.vis_knn')


def vis_token_knn(data_dir, subset="train", k=5, display_k=3, metric="cosine",
                  global_neighbor=False, neighbor_subset="train", context=10):
    if subset == "train":
        display_k = max(2,
                        display_k)  # since neighbor of sample in train is most likely itself, so we want to see rank2 sentence.

    dictionary = TranslationTask.load_dictionary(dictionary_path(data_dir))
    dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, subset), dictionary
    )

    src_sent_offsets = np.cumsum(dataset.sizes)
    src_sent_offsets = np.insert(src_sent_offsets, 0, 0)
    token_num = src_sent_offsets[-1]

    nds = DataStore.from_pretrained(dstore_path(data_dir, neighbor_subset), no_load_keys=True)

    npath = neighbor_path(data_dir, subset, k)
    print(f"Using token neighbor file at {npath}")
    neighbors = np.memmap(npath,
                          dtype=np.int64, mode='r',
                          shape=(token_num, k))

    while True:
        try:
            bpe_symbol = "@@ "
            sent_idx = int(input(f"sent id(0-{len(dataset)}): ").strip())
            # sent_idx = int(input(f"token offset(0-{token_num}): ").strip())
            sent_token_ids = dataset[sent_idx][: 128]
            sent_offset = src_sent_offsets[sent_idx]
            # print(colored(f"origin sent: {dictionary.string(sent_token_ids, bpe_symbol=bpe_symbol)} |||  {' '.join([dictionary[x] for x in sent_token_ids])}", 'red'))
            print(colored(f"origin sent: {' '.join([dictionary[x] for x in sent_token_ids])}", 'red'))
            for token_offset, token_id in enumerate(sent_token_ids):
                offset = sent_offset + token_offset
                for rank in range(min(k, display_k)):
                    neighbor_offset = neighbors[offset][rank]
                    if neighbor_offset == -1:
                        continue
                    if subset == "train" and neighbor_offset == offset:
                        continue
                    neighbor_context_start = max(0, neighbor_offset - context)
                    neighbor_context_end = min(nds.dstore_size, neighbor_offset + context)
                    neighbor_context = nds.vals[neighbor_context_start: neighbor_context_end]
                    neighbor_colored_idx = neighbor_offset - neighbor_context_start
                    print(f"neighbor of token {dictionary[token_id]}({token_id})@{rank}: {' '.join([dictionary[x] if idx != neighbor_colored_idx else colored(dictionary[x], 'red') for idx, x in enumerate(neighbor_context)])}")

        except Exception as e:
            LOGGING.error("error", exc_info=True)


def generate_val(data_dir, subset):
    """generate vals.npy"""
    dictionary = TranslationTask.load_dictionary(dictionary_path(data_dir))
    dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, subset), dictionary
    )

    src_sent_offsets = np.cumsum(dataset.sizes)
    src_sent_offsets = np.insert(src_sent_offsets, 0, 0)
    token_num = src_sent_offsets[-1]
    val_file = value_path(data_dir, subset)
    array = np.memmap(val_file, dtype=np.int32, shape=(token_num, 1), mode="w+")
    for sent_idx in range(len(dataset)):
        array[src_sent_offsets[sent_idx]: src_sent_offsets[sent_idx+1]] = dataset[sent_idx][:, None]


if __name__ == '__main__':

    # for subset in ["train", "valid", "test"]:
    #     generate_val(
    #         data_dir="/userhome/yuxian/data/lm/wiki-103/data-bin",
    #         subset=subset
    #     )

    vis_token_knn(
        # data_dir="/data/yuxian/datasets/wikitext-103/data-bin",
        data_dir="/userhome/yuxian/data/lm/wiki-103/data-bin",
        # data_dir="/userhome/yuxian/data/lm/one-billion/data-bin-256",
        # data_dir="/data/yuxian/wiki103-yunnao/data-bin",
        # data_dir="/userhome/yuxian/data/lm/enwik8/data-bin",
        subset="test",
        k=128,
        metric="cosine",
        display_k=3,
    )
