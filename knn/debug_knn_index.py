# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/8/16 14:15
@desc: 

"""

from knn.knn_model import KNNModel

import logging


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGING = logging.getLogger('knn.find_knn')


knn = KNNModel(
    index_file="/userhome/yuxian/data/lm/one-billion/data-bin-256/train_dstore/faiss_store.cosine",
    dstore_dir="/userhome/yuxian/data/lm/one-billion/data-bin-256/train_dstore",
    no_load_keys=True,
    use_memory=False,
    cuda=0
)
