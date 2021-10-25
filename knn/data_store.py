# encoding: utf-8
"""



@version: 1.0
@file: data_store
"""

import os
import numpy as np
import time
import json

import logging


LOGGING = logging.getLogger(__name__)


class DataStore:
    """
    DataStore to save hidden states
    Attributes:
        keys: [dstore_size, hidden_size]
        vals: [dstore_size, 2], sent-idx and token-idx
    """

    def __init__(self, dstore_size: int, hidden_size: int, dstore_dir: str, vocab_size: int = None, mode="r",
                 dstore_fp16: bool = False, no_load_keys: bool = False, use_memory: bool = False,
                 val_size: int = 2):
        self.dstore_size = dstore_size
        self.hidden_size = hidden_size
        self.dstore_dir = dstore_dir
        self.vocab_size = vocab_size
        self.no_load_keys = no_load_keys
        self.dstore_fp16 = dstore_fp16
        self.val_size = val_size

        os.makedirs(dstore_dir, exist_ok=True)
        if not no_load_keys:
            key_file = os.path.join(dstore_dir, "keys.npy")

            self.keys = np.memmap(key_file,
                                  dtype=np.float16 if self.dstore_fp16 else np.float32,
                                  mode=mode,
                                  shape=(dstore_size, hidden_size))
        val_file = os.path.join(dstore_dir, "vals.npy")
        self.vals = np.memmap(val_file,
                              dtype=np.int16 if self.dstore_fp16 and self.vocab_size < 2**15 else np.int32,
                              mode=mode,
                              shape=(dstore_size, val_size))
        if self.val_size == 1:
            self.vals = self.vals.reshape(-1)

        if use_memory and mode == "r":
            start = time.time()

            if not self.no_load_keys:
                self.memory_keys = np.zeros((self.dstore_size, self.hidden_size), dtype=self.keys.dtype)
                self.memory_keys = self.keys[:]
                self.keys = self.memory_keys
            self.vals = np.array(self.vals)
            LOGGING.debug('Loading to memory took {} s'.format(time.time() - start))

    def save_info(self):
        """save information of datastore"""
        json.dump(self.info, open(os.path.join(self.dstore_dir, "info.json"), "w"),
                  sort_keys=True, indent=4, ensure_ascii=False)

    @staticmethod
    def exists(dstore_dir):
        return (
            os.path.exists(os.path.join(dstore_dir, "keys.npy")) and
            os.path.exists(os.path.join(dstore_dir, "vals.npy"))
        )

    @property
    def info(self):
        """get info"""
        info = {
            "dstore_size": self.dstore_size,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "dstore_fp16": self.dstore_fp16,
            "val_size": self.val_size
        }
        return info

    @classmethod
    def from_pretrained(cls, dstore_dir: str, no_load_keys=False, use_memory=False, mode="r"):
        """load DataStore from pretrained file"""
        info = json.load(open(os.path.join(dstore_dir, "info.json")))
        dstore_size, hidden_size, vocab_size, dstore_fp16, val_size = (
            info["dstore_size"],
            info["hidden_size"],
            info.get("vocab_size", None),
            info.get("dstore_fp16", False),
            info.get("val_size", 1),
        )
        return cls(dstore_size=dstore_size, hidden_size=hidden_size, dstore_dir=dstore_dir, dstore_fp16=dstore_fp16,
                   vocab_size=vocab_size, no_load_keys=no_load_keys, mode=mode, use_memory=use_memory, val_size=val_size)


if __name__ == '__main__':
    # dstore_dir = "/data/yuxian/datasets/wikitext-103/data-bin/test_dstore"
    dstore_dir = "/userhome/yuxian/data/lm/one-billion/data-bin-256/test_dstore"
    ds = DataStore.from_pretrained(dstore_dir)
    print(ds.info)
    print(ds.keys[-10:])
    print(ds.vals[-10:])
    #
    # import numpy as np
    # import torch
    #
    #
    # def mem():
    #     import os, psutil
    #     return psutil.Process(os.getpid()).memory_info().rss // 1024
    #
    # mem_base = mem()
    # for _ in range(10):
    #     torch.LongTensor([x for x in ds.vals[:100000]])
    #     print(mem() - mem_base)