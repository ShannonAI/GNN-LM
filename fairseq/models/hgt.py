# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com
@time: 2021/3/1 15:53
@desc: Heterogeneous Graph Transformer(https://arxiv.org/abs/2003.01332)
"""
import math
from typing import Dict, List, Tuple, Optional

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax
from fairseq.incremental_decoding_utils import with_incremental_state
from torch import Tensor


@with_incremental_state
class HGTLayer(nn.Module):
    """
    Heterogeneous Graph Transformer Layer
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        ntype2idx: Dict[str, int],  # map node type to its id
        etype2idx: Dict[str, int],  # todo change str to tuple[str, str, str]
        # map edge type to its id. Note: could share weights across relationships by assigning different edge types to same id
        n_heads: int,
        dropout=0.2,
        use_norm=True,
        two_stream=False,
        attn_drop=0.2
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ntype2idx = ntype2idx
        self.etype2idx = etype2idx
        self.num_types = len(ntype2idx)
        self.num_relations = len(etype2idx)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        assert out_dim % self.n_heads == 0
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm
        self.two_stream = two_stream

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_drop)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def infer(
        self, G: dgl.DGLHeteroGraph,
        h: Dict[str, torch.Tensor],
        etypes: List[Tuple[str, str, str]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """update latest tgt node only."""
        assert not self.two_stream, "not supported yet"
        saved_states = self._get_input_buffer(incremental_state)
        bsz, hidden = h["tgt"].size()
        device = h["tgt"].device
        max_len = 512  # todo 变成参数

        if not saved_states:
            cur_step = 0
            tgt_idxs = torch.full([bsz], fill_value=cur_step, device=device) + \
                       torch.arange(0, bsz, device=device) * max_len
            buffer = {
                "step": torch.zeros([bsz], dtype=torch.long, device=device)
            }
            # update all nodes of all ntypes
            with G.local_scope():
                node_dict, edge_dict = self.ntype2idx, self.etype2idx

                for ntype in G.ntypes:
                    k_linear = self.k_linears[node_dict[ntype]]
                    v_linear = self.v_linears[node_dict[ntype]]
                    q_linear = self.q_linears[node_dict[ntype]]
                    # [bsz, n_heads, d_k]
                    k = k_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                    q = q_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                    v = v_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                    if ntype == "tgt":  # incremental
                        G.nodes[ntype].data[f"{ntype}_k"] = torch.zeros([G.num_nodes(ntype), self.n_heads, self.d_k], device=device)
                        G.nodes[ntype].data[f"{ntype}_k"][tgt_idxs] = k
                        G.nodes[ntype].data[f"{ntype}_v"] = torch.zeros([G.num_nodes(ntype), self.n_heads, self.d_k], device=device)
                        G.nodes[ntype].data[f"{ntype}_v"][tgt_idxs] = v
                        G.nodes[ntype].data[f"{ntype}_q"] = torch.zeros([G.num_nodes(ntype), self.n_heads, self.d_k], device=device)
                        G.nodes[ntype].data[f"{ntype}_q"][tgt_idxs] = q
                    else:
                        G.nodes[ntype].data[f"{ntype}_k"] = k
                        G.nodes[ntype].data[f"{ntype}_v"] = v
                        G.nodes[ntype].data[f"{ntype}_q"] = q

                    # cache k,q,v
                    for att in "kqv":
                       buffer[f"{ntype}_{att}"] = G.nodes[ntype].data[f"{ntype}_{att}"].view(bsz, G.num_nodes(ntype)//bsz, -1)

                for srctype, etype, dsttype in etypes:
                    sub_graph = G[srctype, etype, dsttype]

                    k = G.nodes[srctype].data[f"{srctype}_k"]
                    q = G.nodes[dsttype].data[f"{dsttype}_q"]
                    v = G.nodes[srctype].data[f"{srctype}_v"]

                    e_id = self.etype2idx[etype]

                    # [n_heads, d_k, d_k]
                    relation_att = self.relation_att[e_id]
                    relation_pri = self.relation_pri[e_id]
                    relation_msg = self.relation_msg[e_id]

                    # [bsz, n_heads, d_k]
                    k = torch.einsum("bij,ijk->bik", k, relation_att)
                    v = torch.einsum("bij,ijk->bik", v, relation_msg)

                    sub_graph.srcdata['k'] = k
                    sub_graph.dstdata['q'] = q
                    sub_graph.srcdata[f'v_{srctype}_{etype}_{dsttype}'] = v

                    sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                    attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                    attn_score = self.attn_drop(edge_softmax(sub_graph, attn_score, norm_by='dst'))

                    sub_graph.edata['t'] = attn_score.unsqueeze(-1)

                G.multi_update_all({
                    (srctype, etype, dsttype): (
                    fn.u_mul_e(f'v_{srctype}_{etype}_{dsttype}', 't', 'm'), fn.sum('m', 't'))
                    for srctype, etype, dsttype in etypes
                }, cross_reducer='mean')

                new_h = {}
                for ntype in G.ntypes:
                    n_id = node_dict[ntype]
                    # alpha = torch.sigmoid(self.skip[n_id])
                    t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                    if ntype == "tgt":
                        t = t[tgt_idxs]
                    trans_out = self.drop(self.a_linears[n_id](t))
                    # trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                    trans_out = trans_out + h[ntype]
                    if self.use_norm:
                        out_feat = self.norms[n_id](trans_out)
                    else:
                        out_feat = trans_out
                    new_h[ntype] = out_feat
                    if ntype == "tgt":
                        pad_feat = torch.zeros([G.num_nodes(ntype), hidden], device=device)
                        pad_feat[tgt_idxs] = out_feat
                        buffer[f"{ntype}_out_feat"] = pad_feat.view(bsz, G.num_nodes(ntype)//bsz, -1)
                    else:
                        buffer[f"{ntype}_out_feat"] = out_feat.view(bsz, G.num_nodes(ntype)//bsz, -1)

                self._set_input_buffer(incremental_state,
                                       buffer=buffer)
                return new_h

        else:
            cur_step_int = saved_states["step"][0] + 1
            tgt_idxs = torch.full([bsz], fill_value=cur_step_int, device=device) + \
                       torch.arange(0, bsz, device=device) * max_len
            buffer = {"step": saved_states["step"]+1}
            with G.local_scope():
                node_dict, edge_dict = self.ntype2idx, self.etype2idx

                for ntype in G.ntypes:
                    # copy src/tgt/ntgt feat from cache
                    G.nodes[ntype].data[f"{ntype}_k"] = saved_states[f"{ntype}_k"].view(-1, self.n_heads, self.d_k)
                    G.nodes[ntype].data[f"{ntype}_v"] = saved_states[f"{ntype}_v"].view(-1, self.n_heads, self.d_k)
                    G.nodes[ntype].data[f"{ntype}_q"] = saved_states[f"{ntype}_q"].view(-1, self.n_heads, self.d_k)
                    if ntype == "tgt":
                        k_linear = self.k_linears[node_dict[ntype]]
                        v_linear = self.v_linears[node_dict[ntype]]
                        q_linear = self.q_linears[node_dict[ntype]]
                        # [bsz, n_heads, d_k]
                        k = k_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                        q = q_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                        v = v_linear(h[ntype]).view(-1, self.n_heads, self.d_k)

                        G.nodes[ntype].data[f"{ntype}_k"][tgt_idxs] = k
                        G.nodes[ntype].data[f"{ntype}_v"][tgt_idxs] = v
                        G.nodes[ntype].data[f"{ntype}_q"][tgt_idxs] = q

                    # cache k,q,v
                    for att in "kqv":
                        buffer[f"{ntype}_{att}"] = G.nodes[ntype].data[f"{ntype}_{att}"].view(bsz, G.num_nodes(ntype)//bsz, -1)

                for srctype, etype, dsttype in etypes:
                    if dsttype != "tgt":
                        continue
                    sub_graph = G[srctype, etype, dsttype]
                    # prev_subgraph = prev_G[srctype, etype, dsttype]

                    k = G.nodes[srctype].data[f"{srctype}_k"]
                    q = G.nodes[dsttype].data[f"{dsttype}_q"]
                    v = G.nodes[srctype].data[f"{srctype}_v"]

                    e_id = self.etype2idx[etype]

                    # [n_heads, d_k, d_k]
                    relation_att = self.relation_att[e_id]
                    relation_pri = self.relation_pri[e_id]
                    relation_msg = self.relation_msg[e_id]

                    # [bsz, n_heads, d_k]
                    k = torch.einsum("bij,ijk->bik", k, relation_att)
                    v = torch.einsum("bij,ijk->bik", v, relation_msg)

                    sub_graph.srcdata['k'] = k
                    sub_graph.dstdata['q'] = q
                    sub_graph.srcdata[f'v_{srctype}_{etype}_{dsttype}'] = v

                    sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                    attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                    attn_score = self.attn_drop(edge_softmax(sub_graph, attn_score, norm_by='dst'))

                    sub_graph.edata['t'] = attn_score.unsqueeze(-1)

                G.multi_update_all({
                    (srctype, etype, dsttype): (
                    fn.u_mul_e(f'v_{srctype}_{etype}_{dsttype}', 't', 'm'), fn.sum('m', 't'))
                    for srctype, etype, dsttype in etypes if dsttype == "tgt"
                }, cross_reducer='mean')  # todo(yuxian): check failure of 0 in-degree nodes of subgraph

                # only update a subgraph
                # subgraph_nodes = {
                #     ntype: torch.ones(G.num_nodes(ntype, )).bool()
                #     for ntype in ["src", "ntgt"]  # we ignore nsrc, since it has not edges pointing to tgt
                # }
                # subgraph_nodes["tgt"] = tgt_idxs
                # subG = dgl.node_subgraph(G, nodes=subgraph_nodes)
                #
                # subG.multi_update_all({
                #     (srctype, etype, dsttype): (
                #     fn.u_mul_e(f'v_{srctype}_{etype}_{dsttype}', 't', 'm'), fn.sum('m', 't'))
                #     for srctype, etype, dsttype in etypes if dsttype == "tgt"
                # }, cross_reducer='mean')

                new_h = {}
                for ntype in G.ntypes:
                    if ntype == "tgt":
                        n_id = node_dict[ntype]
                        # alpha = torch.sigmoid(self.skip[n_id])
                        t = G.nodes[ntype].data['t'].view(-1, self.out_dim)[tgt_idxs]
                        # t = subG.nodes[ntype].data['t'].view(-1, self.out_dim)
                        trans_out = self.drop(self.a_linears[n_id](t))
                        # trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                        trans_out = trans_out + h[ntype]
                        if self.use_norm:
                            out_feat = self.norms[n_id](trans_out)
                        else:
                            out_feat = trans_out
                        # cache_g.nodes[ntype].data["out_feat"] = prev_G.nodes[ntype].data["out_feat"]
                        new_out = saved_states[f"{ntype}_out_feat"].view(-1, hidden)
                        new_out[tgt_idxs] = out_feat
                        buffer[f"{ntype}_out_feat"] = new_out.view(bsz, G.num_nodes(ntype)//bsz, -1)
                    else:
                        # out_feat = prev_G.nodes[ntype].data["out_feat"]
                        out_feat = saved_states[f"{ntype}_out_feat"].view(-1, hidden)
                        buffer[f"{ntype}_out_feat"] = out_feat.view(bsz, G.num_nodes(ntype)//bsz, -1)
                    new_h[ntype] = out_feat

                self._set_input_buffer(incremental_state,
                                       buffer=buffer)

                return new_h

    def forward(
        self, G: dgl.DGLHeteroGraph,
        h: Dict[str, torch.Tensor],
        etypes: List[Tuple[str, str, str]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
    ) -> Dict[str, torch.Tensor]:

        etypes = etypes or G.canonical_etypes

        incremental = incremental_state is not None
        if incremental:
            return self.infer(G, h, etypes, incremental_state)

        with G.local_scope():
            node_dict, edge_dict = self.ntype2idx, self.etype2idx

            for ntype in G.ntypes:
                k_linear = self.k_linears[node_dict[ntype]]
                v_linear = self.v_linears[node_dict[ntype]]
                q_linear = self.q_linears[node_dict[ntype]]
                # [bsz, n_heads, d_k]
                G.nodes[ntype].data[f"{ntype}_k"] = k_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                G.nodes[ntype].data[f"{ntype}_v"] = v_linear(h[ntype]).view(-1, self.n_heads, self.d_k)
                G.nodes[ntype].data[f"{ntype}_q"] = q_linear(h[ntype]).view(-1, self.n_heads, self.d_k)

            if self.two_stream:
                q_linear = self.q_linears[node_dict["tgt"]]
                h_tgt_tilde = h["tgt_tilde"] if "tgt_tilde" in h else h["tgt"]
                # [bsz, n_heads, d_k]
                G.nodes["tgt"].data[f"tgt_tilde_q"] = q_linear(h_tgt_tilde).view(-1, self.n_heads, self.d_k)
                G.nodes["tgt"].data[f"tgt_tilde_k"] = k_linear(h_tgt_tilde).view(-1, self.n_heads, self.d_k)
                G.nodes["tgt"].data[f"tgt_tilde_v"] = v_linear(h_tgt_tilde).view(-1, self.n_heads, self.d_k)

            for srctype, etype, dsttype in etypes:
                sub_graph = G[srctype, etype, dsttype]

                k = G.nodes[srctype].data[f"{srctype}_k"]
                q = G.nodes[dsttype].data[f"{dsttype}_q"]
                v = G.nodes[srctype].data[f"{srctype}_v"]

                e_id = self.etype2idx[etype]

                # [n_heads, d_k, d_k]
                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                # [bsz, n_heads, d_k]
                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{srctype}_{etype}_{dsttype}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = self.attn_drop(edge_softmax(sub_graph, attn_score, norm_by='dst'))

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

                if self.two_stream and (srctype, etype, dsttype) in [("tgt", "intra", "tgt"),
                                                                     ("src", "intra", "tgt")]:
                    q = G.nodes[dsttype].data["tgt_tilde_q"]

                    e_id = self.etype2idx[etype]

                    # [n_heads, d_k, d_k]
                    relation_pri = self.relation_pri[e_id]
                    sub_graph.dstdata['q_tilde'] = q

                    # to prevent information leakage
                    if (srctype, etype, dsttype) == ("src", "intra", "tgt"):
                        sub_graph.apply_edges(fn.v_dot_u('q_tilde', 'k', 't_tilde'))
                    else:
                        u, v = sub_graph.edges(form='uv', order='eid', etype=("tgt", "intra", "tgt"))
                        self_loop_mask = u == v  # [nedges]
                        sub_graph.apply_edges(fn.v_dot_u('q_tilde', 'k', 't_tilde'), edges=torch.where(~self_loop_mask)[0])
                        sub_graph.apply_edges(fn.v_dot_u('q_tilde', 'k_tilde', 't_tilde'), edges=torch.where(self_loop_mask)[0])
                    # [nedges, nheads]
                    attn_score = sub_graph.edata.pop('t_tilde').sum(-1) * relation_pri / self.sqrt_dk
                    attn_score = self.attn_drop(edge_softmax(sub_graph, attn_score, norm_by='dst'))
                    sub_graph.edata['t_tilde'] = attn_score.unsqueeze(-1)

            G.multi_update_all({
                (srctype, etype, dsttype): (fn.u_mul_e(f'v_{srctype}_{etype}_{dsttype}', 't', 'm'), fn.sum('m', 't'))
                for srctype, etype, dsttype in etypes
            }, cross_reducer='mean')

            if self.two_stream:
                # todo tgt-intra-tgt should use v_tilde for self-loop

                G.update_all(fn.u_mul_e(f'v_tgt_intra_tgt', 't_tilde', 'm'), fn.sum('m', 't_intra_tgt'),
                             etype=('tgt', 'intra', 'tgt'))
                G.update_all(fn.u_mul_e(f'v_src_intra_tgt', 't_tilde', 'm'), fn.sum('m', 't_intra_src'),
                             etype=('src', 'intra', 'tgt'))

            new_h = {}
            for ntype in G.ntypes:  # todo use more FFN like transformer?
                n_id = node_dict[ntype]
                # alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                # trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                trans_out = trans_out + h[ntype]
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
                # two-stream
                if ntype == "tgt" and self.two_stream:
                    n_id = node_dict["tgt"]
                    t = (G.nodes[ntype].data['t_intra_tgt'].view(-1, self.out_dim) +
                         G.nodes[ntype].data['t_intra_src'].view(-1, self.out_dim)) / 2
                    trans_out = self.drop(self.a_linears[n_id](t))
                    trans_out = trans_out + (h["tgt_tilde"] if "tgt_tilde" in h else h["tgt"])
                    if self.use_norm:
                        new_h["tgt_tilde"] = self.norms[n_id](trans_out)
                    else:
                        new_h["tgt_tilde"] = trans_out

            return new_h

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if isinstance(input_buffer_k, torch.Tensor):
                        input_buffer[k] = input_buffer_k.index_select(0, new_order)
                    elif isinstance(input_buffer_k, dgl.DGLGraph):
                        graphs = dgl.unbatch(input_buffer_k)
                        input_buffer[k] = dgl.batch([graphs[i] for i in new_order])
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "prev_g")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "prev_g", buffer)


class HGT(nn.Module):
    def __init__(
        self,
        ntype2idx: Dict[str, int],
        etype2idx: Dict[str, int],
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        n_heads: int,
        use_norm: bool = True,
        dropout: float = 0.0,
        two_stream: bool = False,
        attn_drop: float = 0.0
    ):
        super(HGT, self).__init__()
        self.ntype2idx = ntype2idx
        self.etype2idx = etype2idx
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        if self.in_dim != self.hidden_dim:
            for t in range(len(ntype2idx)):
                self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(hidden_dim, hidden_dim,
                                     ntype2idx, etype2idx, n_heads,
                                     use_norm=use_norm, dropout=dropout, two_stream=two_stream,
                                     attn_drop=attn_drop))
        if hidden_dim != out_dim:
            self.out = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, G: dgl.DGLHeteroGraph,
        features: Dict[str, torch.Tensor] = None,
        etypes: List[Tuple[str, str, str]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None
    ):
        h = {}
        for ntype in G.ntypes:
            input_feat = None if not features else features.get(ntype)
            if input_feat is None:
                input_feat = G.nodes[ntype].data["h"]
            if self.in_dim != self.hidden_dim:
                n_id = self.ntype2idx[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](input_feat))
            else:
                h[ntype] = input_feat

        for i in range(self.n_layers):
            h = self.gcs[i](G, h, etypes=etypes, incremental_state=incremental_state)
        return {k: self.out(h[k]) for k in h} if self.hidden_dim != self.out_dim else h


if __name__ == '__main__':
    def test_hgt():
        g = dgl.heterograph({
            ("src", "intra", "src"): ([0, 0, 1, 1], [0, 1, 0, 1]),
            ("src", "intra", "tgt"): ([0, 0, 1, 1], [0, 1, 0, 1]),
            ("tgt", "intra", "tgt"): ([0, 0, 1], [0, 1, 1]),

            ('nsrc', 'inter', 'src'): ([0], [0]),
            ('src', 'inter', 'nsrc'): ([0], [0]),
            ('ntgt', 'inter', 'tgt'): ([0], [0]),

            ('ntgt', 'intra', 'ntgt'): ([0], [0]),
            ('ntgt', 'intra', 'nsrc'): ([0], [0]),
            ('nsrc', 'intra', 'ntgt'): ([0], [0]),
            ('nsrc', 'intra', 'nsrc'): ([0], [0]),
        })

        hidden_dim = 128

        g.nodes["src"].data["h"] = torch.rand([g.num_nodes("src"), hidden_dim]) - 0.5
        g.nodes["tgt"].data["h"] = torch.rand([g.num_nodes("tgt"), hidden_dim]) - 0.5
        g.nodes["nsrc"].data["h"] = torch.rand([g.num_nodes("nsrc"), hidden_dim]) - 0.5
        g.nodes["ntgt"].data["h"] = torch.rand([g.num_nodes("ntgt"), hidden_dim]) - 0.5

        g = g.to('cuda:0')
        model = HGT(
            etype2idx={"intra": 0, "inter": 1},
            ntype2idx = {"src": 0, "nsrc": 1, "tgt": 2, "ntgt": 3},
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=3, n_heads=2,
        ).to('cuda:0')
        # y = model(g)
        # print(y.shape)
        from tqdm import tqdm
        for _ in tqdm(range(100000)):
            y = model(g, features={ntype: g.nodes[ntype].data["h"] for ntype in g.ntypes})


    test_hgt()