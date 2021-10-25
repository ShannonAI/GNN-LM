# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Dict

import numpy as np
import pyarrow.plasma as plasma
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MonolingualDataset,
    GraphMonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    GraphTokenBlockDataset,
    MmapDataset,
    TruncateDataset,
    TruncatedDictionary,
)
# from fairseq.data.plasma_utils import PlasmaArray
from fairseq.data.new_plasma_utils import PlasmaArray
from fairseq.tasks import FairseqTask, register_task
from knn.path_utils import *

logger = logging.getLogger(__name__)


@register_task("language_modeling")
class LanguageModelingTask(FairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')
        parser.add_argument('--max-target-positions', type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='truncate sequences to --tokens-per-sample')
        ## knnlm related items
        parser.add_argument('--knn-keytype', type=str, default=None,
                            help='for knnlm WT103 results, use last_ffn_input')
        parser.add_argument('--probe', default=8, type=int,
                            help='for FAISS, the number of lists to query')
        parser.add_argument('--k', default=1024, type=int,
                            help='number of nearest neighbors to retrieve')
        parser.add_argument('--dstore-size', default=103227021, type=int,
                            help='number of items in the knnlm datastore')
        parser.add_argument('--dstore-filename', type=str, default=None,
                            help='File where the knnlm datastore is saved')
        parser.add_argument('--indexfile', type=str, default=None,
                            help='File containing the index built using faiss for knn')
        parser.add_argument('--lmbda', default=0.0, type=float,
                            help='controls interpolation with knn, 0.0 = no knn')
        parser.add_argument('--knn-sim-func', default="do_not_recomp_ip", type=str,
                            help='similarity function to use for knns')
        parser.add_argument('--faiss-metric-type', default='l2', type=str,
                            help='the distance metric for faiss')
        parser.add_argument('--no-load-keys', default=False, action='store_true',
                            help='do not load keys')
        parser.add_argument('--dstore-fp16', default=False, action='store_true',
                            help='if true, datastore items are saved in fp16 and int16')
        parser.add_argument('--move-dstore-to-mem', default=False, action='store_true',
                            help='move the keys and values for knn to memory')
        parser.add_argument('--load-neighbor', default=False, action='store_true',
                            help='if true, load neighbor information')
        # knnlm related items

        # graph related items
        parser.add_argument('--graph', default=False, action='store_true',
                            help='if true, use graph dataset/label')
        parser.add_argument('--neighbor-context', default='(2, 2)', metavar="B",
                            help='neighbor context, left window size and right window size')
        parser.add_argument('--use-precompute-feat', default=False, action='store_true',
                            help='if true, use pre-computed feat instead of online computed feature'
                                 'this is usefule when context in training is smaller than origin version')
        parser.add_argument('--invalid-neighbor-context', default=1536, type=int,
                            help='invalid neighbor context, e.g. we do not want the neighbor with ground truth be '
                                 'in the origin sentence during training.')
        parser.add_argument('--gcn-k', default=1024, type=int,
                            help='number of nearest neighbors used for gcn')
        parser.add_argument('--dstore-dir', type=str, default=None,
                            help='File where the knnlm datastore is saved')
        parser.add_argument('--index-file', type=str, default=None,
                            help='File containing the index built using faiss for knn')
        parser.add_argument("--plasma_path", type=str, default="",
                            help="load quantized feature to memory")
        parser.add_argument('--gcn-context-window', default=0, type=int,
                            help='context window used for evaluating')
        parser.add_argument('--intra-context', default=0, type=int,
                            help='intra context length')
        parser.add_argument('--deprecated', default=False, action='store_true',
                            help='if true, use deprecated graph build')
        parser.add_argument('--reinit-nfeat', default=False, action='store_true',
                            help='if true, use word embedding as neighbor node feature instead of cached '
                                 'representation')
        # fmt: on

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary
        self.args = args

        if targets is None:
            targets = ["future"]
        self.targets = targets
        self.graph = args.graph
        self.plasma_client = plasma.connect(self.args.plasma_path) if self.args.plasma_path else None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = args.data.split(os.pathsep)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )

        # upgrade old checkpoints
        if hasattr(args, "exclude_self_target"):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(os.pathsep)
        assert len(paths) > 0

        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        if self.args.truncate_sequence:
            dataset = TruncateDataset(dataset, self.args.tokens_per_sample)

        if not self.graph:
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode=self.args.sample_break_mode,
                include_targets=True,
            )

            add_eos_for_other_targets = (
                self.args.sample_break_mode is not None
                and self.args.sample_break_mode != "none"
            )

            self.datasets[split] = MonolingualDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.output_dictionary,
                add_eos_for_other_targets=add_eos_for_other_targets,
                shuffle=False if hasattr(self.args, 'lm_eval') and self.args.lm_eval else True,
                targets=self.targets,
                add_bos_token=self.args.add_bos_token,
            )
        else:
            num_tokens = sum(dataset.sizes)
            neighbor_info = json.load(open(os.path.join(dstore_path(data_path, "train"), "info.json")))
            info = json.load(open(os.path.join(dstore_path(data_path, split), "info.json")))
            num_neighbor_tokens = neighbor_info["dstore_size"]
            precompute_feat_dtype = np.float16 if info["dstore_fp16"] else np.float32
            precompute_feat_size = info["hidden_size"]
            neighbor_tokens_dtype = np.int16 if info["dstore_fp16"] and len(self.dictionary) < 2**15 else np.int32
            if not self.args.reinit_nfeat:
                quant_nfeat_path = quantized_feature_path(data_path, "train")
                quantize_neighbor_feat = (self.load_plasma_array(feat_file=quant_nfeat_path, subset="train") or
                                          PlasmaArray(np.load(quantized_feature_path(data_path, "train"))))
            else:
                quantize_neighbor_feat = None
            dataset = GraphTokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode=self.args.sample_break_mode,
                include_targets=True,

                neighbor_offsets=MmapDataset(neighbor_path(data_dir=data_path, mode=split, k=self.args.gcn_k),
                                             dtype=np.int64, shape=(num_tokens, self.args.gcn_k), warmup=False),
                
                neighbor_tokens=MmapDataset(value_path(data_dir=data_path, mode="train"),
                                            dtype=neighbor_tokens_dtype,
                                            shape=(num_neighbor_tokens, 1), warmup=False),
                quant_neighbor_feats=quantize_neighbor_feat,
                neighbor_context=eval(self.args.neighbor_context),
                precompute_feats=None if not self.args.use_precompute_feat else MmapDataset(
                    feature_path(data_dir=data_path, mode=split),
                    dtype=precompute_feat_dtype, shape=(num_tokens, precompute_feat_size), warmup=False),
                invalid_neighbor_context=self.args.invalid_neighbor_context if split == "train" else 0,
                context_window=self.args.gcn_context_window,
                intra_context=self.args.intra_context,
                deprecated=self.args.deprecated
            )

            add_eos_for_other_targets = (
                self.args.sample_break_mode is not None
                and self.args.sample_break_mode != "none"
            )

            self.datasets[split] = GraphMonolingualDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.output_dictionary,
                add_eos_for_other_targets=add_eos_for_other_targets,
                shuffle=False if hasattr(self.args, 'lm_eval') and self.args.lm_eval else True,
                targets=self.targets,
                add_bos_token=self.args.add_bos_token,
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append an eos to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        tgt_dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=None,  # ignored for "eos" break mode
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="eos",
        )
        src_dataset = PrependTokenDataset(
            StripTokenDataset(
                tgt_dataset,
                # remove eos from (end of) target sequence
                self.source_dictionary.eos(),
            ),
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(src_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token,
            )

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary

    def load_plasma_array(self, feat_file, subset="train"):
        if not self.plasma_client:
            return
        plasma_key = PlasmaArray.generate_object_id(feat_file, suffix=subset)
        if self.plasma_client.contains(plasma_key):
            logger.info(f"using existed array of {feat_file}{subset} in plasma server at {self.args.plasma_path}")
            return PlasmaArray(array=None, obj_id=plasma_key, path=self.args.plasma_path)

    def get_large_arrays(self, split="train", epoch=1, combine=False) -> Dict[str, np.array]:
        """get large arrays used in dataset, keys are the corresponding file path"""

        paths = self.args.data.split(os.pathsep)
        assert len(paths) > 0

        data_path = paths[epoch % len(paths)]

        quant_nfeat_path = quantized_feature_path(data_path, "train")
        quantize_neighbor_feat = np.load(quantized_feature_path(data_path, "train"))

        path2array = {quant_nfeat_path: quantize_neighbor_feat}

        return path2array
