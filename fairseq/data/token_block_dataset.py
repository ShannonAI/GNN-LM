# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import dgl
from typing import Tuple, Dict, List
from functools import lru_cache

from fairseq.data import FairseqDataset, plasma_utils


class TokenBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
    """
    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
    ):
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                'Please build Cython components with: `pip install --editable .` '
                'or `python setup.py build_ext --inplace`'
            )

        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else 'none'

        # For "eos" break-mode, block_size is not required parameters.
        if break_mode == "eos" and block_size is None:
            block_size = 0

        slice_indices = _get_slice_indices_fast(sizes, break_mode, block_size, document_sep_len)
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        if break_mode == "eos":
            # much faster version for eos break mode
            block_to_dataset_index = np.stack(
                [
                    np.arange(len(sizes)),  # starting index in dataset
                    np.zeros(
                        len(sizes), dtype=np.long
                    ),  # starting offset within starting index
                    np.arange(len(sizes)),  # ending index in dataset
                ],
                1,
            )
        else:
            block_to_dataset_index = _get_block_to_dataset_index_fast(
                sizes,
                slice_indices,
            )
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

    def attr(self, attr: str, index: int):
        start_ds_idx, _, _ = self.block_to_dataset_index[index]
        return self.dataset.attr(attr, start_ds_idx)

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )

        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]

        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            if s == 0:
                source = torch.cat([item.new([self.eos]), buffer[0 : e - 1]])
                past_target = torch.cat(
                    [item.new([self.pad, self.eos]), buffer[0 : e - 2]]
                )
            else:
                source = buffer[s - 1 : e - 1]
                if s == 1:
                    past_target = torch.cat([item.new([self.eos]), buffer[0 : e - 2]])
                else:
                    past_target = buffer[s - 2 : e - 2]

            return source, item, past_target

        return item

    def __len__(self):
        return len(self.slice_indices)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(
            {
                ds_idx
                for index in indices
                for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
                for ds_idx in range(start_ds_idx, end_ds_idx + 1)
            }
        )


class GraphTokenBlockDataset(TokenBlockDataset):
    """Break a Dataset of tokens into blocks. combined with graph information

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
        neighbor_offsets: np.array # [corpus_len, k]
        precompute_feats: np.array # [corpus_len, d]
        neighbor_tokens: np.array # [neighbor_corpus_len, 1]
        quant_neighbor_feats: np.array # [neighbor_corpus_len, d]
        neighbor_context: int or tuple of int, add neighbor context to graph.
            if is int, left context and righ context are the same. if is an tuple of int,
            is (left_context, right_context)
        invalid_neighbor_context: invalid neighbor context, useful in training to prevent data leakage,
        context_window: int, defaults to 0. provide more context for evaluation
        intra_context: int, intra context length when adding autoregressive edges
    """
    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        neighbor_offsets=None,
        neighbor_tokens=None,
        quant_neighbor_feats=None,
        neighbor_context=1,
        precompute_feats=None,
        invalid_neighbor_context=0,
        context_window=0,
        intra_context=0,
        deprecated=False
    ):
        super(GraphTokenBlockDataset, self).__init__(dataset, sizes, block_size, pad, eos,
                                                     break_mode, include_targets, document_sep_len)
        self.neighbor_offsets = neighbor_offsets  # [corpus_len, k]
        self.neighbor_tokens = neighbor_tokens  # [neighbor_corpus_len, 1]
        cum_sizes = np.cumsum(sizes)
        self.cum_sizes = np.insert(cum_sizes, 0, 0)
        self.k = self.neighbor_offsets.shape[-1]
        self.quant_neighbor_feats = quant_neighbor_feats
        if isinstance(neighbor_context, int):
            self.left_neighbor_context = neighbor_context
            self.right_neighbor_context = neighbor_context
        else:
            self.left_neighbor_context, self.right_neighbor_context = neighbor_context

        self.precompute_feats = precompute_feats
        self.block_size = block_size
        self.invalid_neighbor_context = invalid_neighbor_context
        self.context_window = context_window
        assert self.include_targets
        self.max_intra_context = intra_context
        self.deprecated = deprecated

    def get_basic_info(self, index):
        if self.context_window == 0 or index == 0:
            start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

            buffer = torch.cat(
                [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
            )
            offsets = torch.cat(
                [self.cum_sizes[idx] + torch.arange(0, len(self.dataset[idx])) for idx in
                 range(start_ds_idx, end_ds_idx + 1)]
            )

            slice_s, slice_e = self.slice_indices[index]
            length = slice_e - slice_s
            s, e = start_offset, start_offset + length
            item = buffer[s:e]
            offsets = offsets[s: e]
            return buffer, item, offsets, s, e, 0
        else:
            start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
            prev_start_ds_idx, prev_start_offset, prev_end_ds_idx = self.block_to_dataset_index[index-1]
            assert prev_end_ds_idx == start_ds_idx or prev_end_ds_idx == start_ds_idx-1
            concat_buffer = torch.cat(
                [self.dataset[idx] for idx in range(prev_start_ds_idx, end_ds_idx + 1)]
            )
            concat_offsets = torch.cat(
                [self.cum_sizes[idx] + torch.arange(0, len(self.dataset[idx])) for idx in
                 range(prev_start_ds_idx, end_ds_idx + 1)]
            )

            slice_s, slice_e = self.slice_indices[index]
            prev_slice_s, prev_slice_e = self.slice_indices[index-1]
            length = slice_e - slice_s
            s, e = start_offset, start_offset + length  # origin s, origin_e, offset based on cur block
            ds_offset = self.cum_sizes[start_ds_idx] - self.cum_sizes[prev_start_ds_idx]
            s, e = s + ds_offset, e + ds_offset  # origin s, origin_e, offset based on prev block
            context_s = max(0, s-self.context_window)
            item = concat_buffer[context_s:e]
            concat_offsets = concat_offsets[context_s: e]
            return concat_buffer, item, concat_offsets, context_s, e, s-context_s

    def __getitem__(self, index):
        # start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        #
        # buffer = torch.cat(
        #     [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        # )
        # offsets = torch.cat(
        #     [self.cum_sizes[idx] + torch.arange(0, len(self.dataset[idx])) for idx in range(start_ds_idx, end_ds_idx + 1)]
        # )
        #
        # slice_s, slice_e = self.slice_indices[index]
        # length = slice_e - slice_s
        # s, e = start_offset, start_offset + length
        # item = buffer[s:e]
        # offsets = offsets[s: e]
        buffer, item, offsets, s, e, loss_start_idx = self.get_basic_info(index)

        if self.include_targets:
            graph_func = self.deprecated_build_graph if self.deprecated else self.new_build_graph
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            neighbor_idxs = self.neighbor_offsets[offsets]
            if s == 0:
                source = torch.cat([item.new([self.eos]), buffer[0: e - 1]])
                # neighbor_idxs = np.concatenate([item.new([-1]*self.k).view(1, self.k),
                #                                 self.neighbor_offsets[offsets[0: e - 1]]])
                graph = graph_func(source, offsets, neighbor_idxs, item)
                past_target = torch.cat(
                    [item.new([self.pad, self.eos]), buffer[0 : e - 2]]
                )
            else:
                source = buffer[s - 1 : e - 1]
                # neighbor_idxs = self.neighbor_offsets[offsets[s - 1: e - 1]]
                graph = graph_func(source, offsets, neighbor_idxs, item)
                if s == 1:
                    past_target = torch.cat([item.new([self.eos]), buffer[0 : e - 2]])
                else:
                    past_target = buffer[s - 2 : e - 2]

            if self.precompute_feats is not None:
                tgt_feats = torch.from_numpy(self.precompute_feats[offsets].astype(np.float32))
                graph.nodes["tgt"].data["h"] = tgt_feats

            return source, item, past_target, graph, torch.LongTensor([loss_start_idx])

        return item

    def __len__(self):
        return len(self.slice_indices)

    def new_build_graph(self, source: torch.Tensor, offsets: np.array, neighbor_idxs:np.array, target: torch.Tensor):
        """
        build dgl graph
        Args:
            source: source tensor
            offsets: np.array
            neighbor_idxs: np.array of shape [len(source), self.k]
            target: target tensor shifted left by 1
        """

        ntgt_id = 0
        tgt2ntgt = [[], []]
        ntgt2ntgt = [[], []]
        ntgt_feats = []
        ntgt_labels = []

        for tgt_idx in range(len(target)):
            # todo: merge same nodes with regard to same tgt_idx
            neighbors = neighbor_idxs[tgt_idx]  # [k]
            for offset in neighbors:
                if offset == -1:
                    continue
                # ignore inner-context tokens in training dataset todo try auto-regressive ignore
                if abs(offsets[tgt_idx] - offset) < self.invalid_neighbor_context:
                    continue
                cur_ntgt_ids = []
                cur_ntgt_offsets = []

                # update ntgt ids/labels/feats
                cur_ntgt_ids.append(ntgt_id)
                cur_ntgt_offsets.append(offset)
                if self.quant_neighbor_feats is not None:
                    ntgt_feats.append(self.quant_neighbor_feats[offset])
                ntgt_labels.append(self.neighbor_tokens[offset])
                tgt2ntgt[0].append(tgt_idx)
                tgt2ntgt[1].append(ntgt_id)
                ntgt_id += 1

                # add context ntgt
                context_offsets = []
                if self.left_neighbor_context:
                    # left context
                    for left_offset in range(max(0, offset-self.left_neighbor_context), offset):
                        context_offsets.append(left_offset)
                if self.right_neighbor_context:
                    # right context
                    for right_offset in range(offset+1, min(len(self.neighbor_offsets.shape[0]),
                                                            offset + 1 + self.right_neighbor_context)):
                        context_offsets.append(right_offset)

                for offset in context_offsets:
                    cur_ntgt_ids.append(ntgt_id)
                    cur_ntgt_offsets.append(offset)
                    ntgt_id += 1
                    if self.quant_neighbor_feats is not None:
                        ntgt_feats.append(self.quant_neighbor_feats[offset])
                    ntgt_labels.append(self.neighbor_tokens[offset])
                cur_ntgt2ntgt = self.build_ntgt_edges(offsets2id={o: ntgt_id for ntgt_id, o in
                                                                  zip(cur_ntgt_ids, cur_ntgt_offsets)},
                                                      context=1,
                                                      bidirect=True)
                ntgt2ntgt[0].extend(cur_ntgt2ntgt[0])
                ntgt2ntgt[1].extend(cur_ntgt2ntgt[1])

        graph = dgl.heterograph({
            ('tgt', 'intra', 'tgt'): self.auto_regressive_edges(len(source), max_context=self.max_intra_context),
            ('ntgt', 'inter', 'tgt'): (torch.LongTensor(tgt2ntgt[1]), torch.LongTensor(tgt2ntgt[0])),
            ('ntgt', 'intra', 'ntgt'): (torch.LongTensor(ntgt2ntgt[0]), torch.LongTensor(ntgt2ntgt[1])),
        })
        if ntgt_feats:
            ntgt_feats = torch.from_numpy(np.stack(ntgt_feats))
            graph.nodes["ntgt"].data["h"] = ntgt_feats
        graph.nodes["ntgt"].data["labels"] = torch.from_numpy(np.concatenate(ntgt_labels))

        return graph

    def deprecated_build_graph(self, source: torch.Tensor, offsets: np.array, neighbor_idxs:np.array, target: torch.Tensor):
        """
        build dgl graph
        NOTE: this graph use ntgt that found by t+1 tgt, which is not valid in inference
        Args:
            source: source tensor
            offsets: np.array
            neighbor_idxs: np.array of shape [len(source), self.k]
            target: target tensor shifted left by 1
        """

        offsets2ntgt_id = {}  # map neighbor tgt nodes to node-ids
        tgt2ntgt = [[], []]
        # ntgt_labels = []  # neighbor tgt node labels
        ntgt_feats = []

        for tgt_idx in range(len(source)):
            neighbors = neighbor_idxs[tgt_idx]  # [k]
            for offset in neighbors:
                if offset == -1:
                    continue
                # ignore inner-context tokens in training dataset
                if abs(offsets[tgt_idx] - offset) < self.invalid_neighbor_context:
                # if self.training and offsets[tgt_idx] > offset - self.block_size:
                    continue

                # update ntgt ids/labels/feats
                if offset not in offsets2ntgt_id:
                    ntgt_id = len(offsets2ntgt_id)
                    offsets2ntgt_id[offset] = ntgt_id
                    # ntgt_labels.append(neighbor_tgt_sent[offset[1]])
                    ntgt_feats.append(self.quant_neighbor_feats[offset])
                else:
                    ntgt_id = offsets2ntgt_id[offset]
                tgt2ntgt[0].append(tgt_idx)
                tgt2ntgt[1].append(ntgt_id)

                # add context ntgt
                context_offsets = []
                if self.left_neighbor_context:
                    # left context
                    for left_offset in range(max(0, offset-self.left_neighbor_context), offset):
                        context_offsets.append(left_offset)
                if self.right_neighbor_context:
                    # right context
                    for right_offset in range(offset+1, min(len(self.neighbor_tokens),
                                                            offset + 1 + self.right_neighbor_context)):
                        context_offsets.append(right_offset)
                for offset in context_offsets:
                    if offset not in offsets2ntgt_id:
                        ntgt_id = len(offsets2ntgt_id)
                        offsets2ntgt_id[offset] = ntgt_id
                        # ntgt_labels.append(self.neighbor_dataset.tgt[offset[0]][offset[1]])
                        ntgt_feats.append(self.quant_neighbor_feats[offset])

        # ntgt->ntgt edges
        ntgt2ntgt = self.build_ntgt_edges(offsets2ntgt_id, 1, bidirect=True)

        graph = dgl.heterograph({
            ('tgt', 'intra', 'tgt'): self.auto_regressive_edges(len(source), max_context=self.max_intra_context),
            ('ntgt', 'inter', 'tgt'): (torch.LongTensor(tgt2ntgt[1]), torch.LongTensor(tgt2ntgt[0])),
            ('ntgt', 'intra', 'ntgt'): (torch.LongTensor(ntgt2ntgt[0]), torch.LongTensor(ntgt2ntgt[1])),
        })
        ntgt_feats = torch.from_numpy(np.stack(ntgt_feats))
        graph.nodes["ntgt"].data["h"] = ntgt_feats

        return graph

    def build_graph(self, source: torch.Tensor, offsets: np.array, neighbor_idxs:np.array, target: torch.Tensor):
        """
        build dgl graph
        Args:
            source: source tensor
            offsets: np.array
            neighbor_idxs: np.array of shape [len(source), self.k]
            target: target tensor shifted left by 1
        """

        offsets2ntgt_id = {}  # map neighbor tgt nodes to node-ids
        tgt2ntgt = [[], []]
        ntgt_feats = []

        for tgt_idx in range(len(source)):
            neighbors = neighbor_idxs[tgt_idx]  # [k]
            for offset in neighbors:
                if offset == -1:
                    continue
                # ignore inner-context tokens in training dataset
                if abs(offsets[tgt_idx] - offset) < self.invalid_neighbor_context:
                    continue

                # update ntgt ids/labels/feats
                if offset not in offsets2ntgt_id:
                    ntgt_id = len(offsets2ntgt_id)
                    offsets2ntgt_id[offset] = ntgt_id
                    ntgt_feats.append(self.quant_neighbor_feats[offset])
                else:
                    ntgt_id = offsets2ntgt_id[offset]
                tgt2ntgt[0].append(tgt_idx)
                tgt2ntgt[1].append(ntgt_id)

                # add context ntgt
                context_offsets = []
                if self.left_neighbor_context:
                    # left context
                    for left_offset in range(max(0, offset-self.left_neighbor_context), offset):
                        context_offsets.append(left_offset)
                if self.right_neighbor_context:
                    # right context
                    for right_offset in range(offset+1, min(len(self.neighbor_tokens),
                                                            offset + 1 + self.right_neighbor_context)):
                        context_offsets.append(right_offset)
                for offset in context_offsets:
                    if offset not in offsets2ntgt_id:
                        ntgt_id = len(offsets2ntgt_id)
                        offsets2ntgt_id[offset] = ntgt_id
                        ntgt_feats.append(self.quant_neighbor_feats[offset])

        # ntgt->ntgt edges
        # ntgt_l2r = self.build_ntgt_edges(offsets2ntgt_id, self.neighbor_context, bidirect=False)
        ntgt_l2r = self.build_ntgt_edges(offsets2ntgt_id, 1, bidirect=True)  # todo: set it in args

        graph = dgl.heterograph({
            ('tgt', 'intra', 'tgt'): self.auto_regressive_edges(len(source), max_context=self.max_intra_context),
            ('ntgt', 'inter', 'tgt'): (torch.LongTensor(tgt2ntgt[1]), torch.LongTensor(tgt2ntgt[0])),
            ('ntgt', 'intra', 'ntgt'): (torch.LongTensor(ntgt_l2r[1]), torch.LongTensor(ntgt_l2r[0])),
        })
        ntgt_feats = torch.from_numpy(np.stack(ntgt_feats))
        graph.nodes["ntgt"].data["h"] = ntgt_feats

        return graph

    @staticmethod
    def build_ntgt_edges(offsets2id: Dict[int, int], context: int = 0, bidirect=False) -> Tuple[List[int], List[int]]:
        """
        add edges of ntgt nodes that within context range
        Examples:
            >>> offsets2id={0: 0, 1: 1, 2: 2, 12:3, 13: 4}
            >>> GraphTokenBlockDataset.build_ntgt_edges(offsets2id, 3)
            ([0, 0, 1, 0, 1, 2, 3, 3, 4], [0, 1, 1, 2, 2, 2, 3, 4, 4])
            >>> GraphTokenBlockDataset.build_ntgt_edges(offsets2id, 0)
            ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        """

        if not offsets2id:
            return [], []

        nodes = [(nid, offset) for offset, nid in offsets2id.items()]
        nodes = sorted(nodes, key=lambda x: x[1])
        src = []
        tgt = []

        start = 0
        end = -1
        length = len(nodes)
        while start < length:
            # find right until reach out context
            while end+1 < length and nodes[end+1][1] <= nodes[start][1] + context:
                end += 1
                tgt_node = nodes[end]
                for s in range(start, end+1):
                    src.append(nodes[s][0])
                    tgt.append(tgt_node[0])
            start += 1
        if bidirect:
            l = len(src)
            for idx in range(l):
                s, t = src[idx], tgt[idx]
                if s != t:
                    src.append(t)
                    tgt.append(s)
        return src, tgt

    @staticmethod
    @lru_cache(maxsize=-1)
    def auto_regressive_edges(length: int, max_context: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        auto_regressive_mask = torch.triu(torch.ones(length, length, dtype=torch.bool))
        if max_context:
            context_mask = ~torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=max_context)
            auto_regressive_mask = torch.logical_and(auto_regressive_mask, context_mask)
        us, vs = torch.where(auto_regressive_mask)
        return us, vs

    @property
    def supports_prefetch(self):
        return False
