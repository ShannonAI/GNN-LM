# encoding: utf-8
"""



@desc: quantize features to make sure that it could fit into RAM

"""

import argparse
import logging

import faiss
import numpy as np
import torch
from tqdm import tqdm

from knn.data_store import DataStore
from knn.path_utils import *
from knn.pq_wrapper import TorchPQCodec

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGING = logging.getLogger('knn.quantize-features')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="path to binary dataset directory")
    parser.add_argument("--prefix", type=str, default="de-en", help="prefix of binary file")
    parser.add_argument("--index", type=str, default="OPQ64_512,PQ64", help="quantizer index")
    parser.add_argument("--subset", type=str, default="train", help="train/valid/test")
    parser.add_argument("--code-size", type=int, default=64, help="bytes of quantized feature")
    parser.add_argument("--chunk-size", type=int, default=10000000, help="maximum number of features to train")
    parser.add_argument("--compute-error", action="store_true", default=False,
                        help="compute reconstruction error")
    parser.add_argument("--use-gpu", action="store_true", default=False,
                        help="use gpu")
    parser.add_argument("--norm", action="store_true", default=False,
                        help="normalize feature vector to unit vector before quantize")
    parser.add_argument("--pretrained_quantizer", action="store_true", default=False,
                        help="use pretrained quantizer to encode features")
    args = parser.parse_args()

    data_dir = args.data_dir
    subset = args.subset
    code_size = args.code_size
    chunk_size = args.chunk_size

    ds = DataStore.from_pretrained(dstore_dir=dstore_path(data_dir=data_dir, subset=subset))

    info = ds.info
    hidden_size = info["hidden_size"]
    total_tokens = info["dstore_size"]

    if args.pretrained_quantizer:
        save_path = quantizer_path(data_dir)
        LOGGING.info(f"load pretrained quantizer at {save_path}")
        quantizer = faiss.read_index(save_path)
        quantizer = TorchPQCodec(quantizer)
        if args.use_gpu:
            LOGGING.info("Using gpu")
            quantizer = quantizer.cuda()
            # we need only a StandardGpuResources per GPU
            # res = faiss.StandardGpuResources()
            # co = faiss.GpuClonerOptions()
            # # here we are using a 64-byte PQ, so we must set the lookup tables to
            # # 16 bit float (this is due to the limited temporary memory).
            # co.useFloat16 = True
            # quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer, co)
    else:
        LOGGING.info(f"Train quantized codes on first {chunk_size} features")
        train_features = np.array(ds.keys[: chunk_size])
        if args.norm:
            norm = np.sqrt(np.sum(train_features ** 2, axis=-1, keepdims=True))
            train_features /= norm

        quantizer = faiss.index_factory(hidden_size, args.index)

        if args.use_gpu:
            LOGGING.info("Using gpu for training")
            # we need only a StandardGpuResources per GPU
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()

            # here we are using a 64-byte PQ, so we must set the lookup tables to
            # 16 bit float (this is due to the limited temporary memory).
            co.useFloat16 = True
            quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer, co)

        LOGGING.info("Training Product Quantizer")
        quantizer.train(train_features.astype(np.float32))

        save_path = quantizer_path(data_dir, norm=args.norm)
        faiss.write_index(quantizer, save_path)
        LOGGING.info(f"Save quantizer to {save_path}")
        quantizer = TorchPQCodec(quantizer)
        if args.use_gpu:
            quantizer = quantizer.cuda()

    quantized_codes = np.zeros([total_tokens, code_size], dtype=np.uint8)

    # encode
    start = 0
    total_error = 0
    pbar = tqdm(total=total_tokens, desc="Computing codes")
    while start < total_tokens:
        end = min(total_tokens, start + chunk_size)
        x = np.array(ds.keys[start: end].astype(np.float32))
        bsz = 8192
        intra_offset = 0
        while intra_offset < end - start:
            batch_x = x[intra_offset: intra_offset + bsz]
            if args.norm:
                norm = np.sqrt(np.sum(batch_x ** 2, axis=-1, keepdims=True))
                batch_x /= norm

            # codes = quantizer.sa_encode(batch_x)
            batch_x = torch.from_numpy(batch_x)
            if args.use_gpu:
                batch_x = batch_x.cuda()
            codes = quantizer.encode(batch_x)

            if args.compute_error:
                # x2 = quantizer.sa_decode(codes)
                x2 = quantizer.decode(codes)
                # compute reconstruction error
                avg_relative_error = (((batch_x - x2)**2).sum() / (batch_x ** 2).sum()).item()
                pbar.set_postfix({"L2 error": avg_relative_error})
                total_error += avg_relative_error * batch_x.shape[0]
            quantized_codes[start+intra_offset: start+intra_offset+batch_x.shape[0]] = codes.cpu()
            intra_offset += batch_x.shape[0]
            pbar.update(batch_x.shape[0])
        start = end

    if args.compute_error:
        LOGGING.info(f"Avg Reconstruction error: {total_error/total_tokens}")

    qt_path = quantized_feature_path(data_dir, subset)
    np.save(qt_path, quantized_codes)
    LOGGING.info(f"Save quantized feature to {qt_path}")


if __name__ == '__main__':
    main()
