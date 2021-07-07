# encoding: utf-8
"""
@desc: https://gist.github.com/mdouze/94bd7a56d912a06ac4719c50fa5b01ac

"""

import faiss
import numpy as np
import torch


class NumpyPQCodec:

    def __init__(self, index, metric="ip"):

        assert index.is_trained
        assert metric in ["l2", "ip"]
        self.metric = metric

        # handle the pretransform
        if isinstance(index, faiss.IndexPreTransform):
            vt = faiss.downcast_VectorTransform(index.chain.at(0))
            assert isinstance(vt, faiss.LinearTransform)
            b = faiss.vector_to_array(vt.b)
            A = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
            self.pre = (A, b)
            index = faiss.downcast_index(index.index)
        else:
            self.pre = None

        # extract the PQ centroids
        assert isinstance(index, faiss.IndexPQ)
        pq = index.pq
        cen = faiss.vector_to_array(pq.centroids)
        cen = cen.reshape(pq.M, pq.ksub, pq.dsub)
        assert pq.nbits == 8
        self.centroids = cen
        self.norm2_centroids = (cen ** 2).sum(axis=2)

        # compute sdc tables in numpy
        if self.metric == "l2":
            cent1 = np.expand_dims(cen, axis=2)  # [M, ksub, 1, dsub]
            cent2 = np.expand_dims(cen, axis=1)  # [M, 1, ksub, dsub]
            sdc_table = -np.sqrt(((cent1 - cent2) ** 2).sum(3))  # [M, ksub, ksub]

        else:
            cen2 = cen.transpose(0, 2, 1)  # [M, dsub, ksub]
            sdc_table = np.matmul(cen, cen2)  # [M, ksub, ksub]
        self.sdc_table = sdc_table

    def encode(self, x):
        if self.pre is not None:
            A, b = self.pre
            x = x @ A.T
            if b.size > 0:
                x += b

        n, d = x.shape
        cen = self.centroids
        M, ksub, dsub = cen.shape
        codes = np.empty((n, M), dtype='uint8')
        # maybe possible to vectorize this loop...
        for m in range(M):
            # compute all per-centroid distances, ignoring the ||x||^2 term
            xslice = x[:, m * dsub:(m + 1) * dsub]
            dis = self.norm2_centroids[m] - 2 * xslice @ cen[m].T
            codes[:, m] = dis.argmin(axis=1)
        return codes

    def decode(self, codes):
        n, MM = codes.shape
        cen = self.centroids
        M, ksub, dsub = cen.shape
        assert MM == M
        x = np.empty((n, M * dsub), dtype='float32')
        for m in range(M):
            xslice = cen[m, codes[:, m]]
            x[:, m * dsub:(m + 1) * dsub] = xslice
        if self.pre is not None:
            A, b = self.pre
            if b.size > 0:
                x -= b
            x = x @ A
        return x


class TorchPQCodec(NumpyPQCodec, torch.nn.Module):

    def __init__(self, index, metric="ip"):
        NumpyPQCodec.__init__(self, index, metric)
        torch.nn.Module.__init__(self)
        # just move everything to torch on the given device
        if self.pre:
            A, b = self.pre
            self.pre_torch = True
            self.register_buffer("A", torch.from_numpy(A))
            self.register_buffer("b", torch.from_numpy(b))
        else:
            self.pre_torch = False
        self.register_buffer("centroids_torch", torch.from_numpy(self.centroids))  # [M, ksub, dsub]
        self.register_buffer("norm2_centroids_torch", torch.from_numpy(self.norm2_centroids))  # [M, ksub]
        self.register_buffer("sdc_table_torch", torch.from_numpy(self.sdc_table))  # [M, ksub, ksub]

    def compute_sim(self, src, tgt):
        """
        Args:
            src: [n, M]  Quantized tensor of torch.uint8
            tgt: [m, M]  Quantized tensor of torch.uint8
        Returns:
            sim: [n, m] float tensor
        """
        sdc_table = self.sdc_table_torch  # [M, ksub, ksub]
        n = src.size(0)
        m = tgt.size(0)
        device = src.device
        M, ksub, _ = sdc_table.size()
        # [1, 1, M, ksub, ksub]
        sdc_table = sdc_table.unsqueeze(0).unsqueeze(0)

        zeros = torch.zeros([1, 1, 1], device=device, dtype=torch.long).expand(n, m, M)
        # [n, m, M]
        subspace_sim = sdc_table[
            zeros,
            zeros,
            torch.arange(M, device=device).unsqueeze(0).unsqueeze(0).expand(n, m, -1),
            src.long().unsqueeze(1).expand(-1, m, -1),
            tgt.long().unsqueeze(0).expand(n, -1, -1),
        ]
        return subspace_sim.sum(-1)  # [n, m]

    def encode(self, x):
        """

        Args:
            x: [n, D], where D = M * dsub. M is the number of subspaces, dsub is the dimension of each subspace

        Returns:
            codes: [n, M]
        """
        if self.pre_torch:
            A, b = self.A, self.b
            x = x @ A.t()
            if b.numel() > 0:
                x += b

        n, d = x.shape
        cen = self.centroids_torch  # [M, ksub, dsub]
        M, ksub, dsub = cen.shape

        # for loop version
        # codes = torch.empty((n, M), dtype=torch.uint8, device=x.device)
        # # maybe possible to vectorize this loop...
        # for m in range(M):
        #     # compute all per-centroid distances, ignoring thecen.shape ||x||^2 term
        #     xslice = x[:, m * dsub:(m + 1) * dsub]  # [n, dsub]
        #     dis = self.norm2_centroids_torch[m] - 2 * xslice @ cen[m].t()
        #     codes[:, m] = dis.argmin(dim=1)

        # parallel version
        x = x.view(n, M, dsub).unsqueeze(-2)  # [n, M, 1, dsub]
        norm = self.norm2_centroids_torch.unsqueeze(0)  # [1, M, ksub]
        cen = cen.transpose(1, 2).unsqueeze(0)  # [1, M, dsub, ksub]
        dot_product = torch.matmul(x, cen).squeeze(-2)  # [n, M, 1, ksub]
        dis = norm - 2 * dot_product  # [n, M, ksub]
        codes = dis.argmin(dim=2).to(torch.uint8)

        return codes

    def decode(self, codes):
        """

        Args:
            codes: [n, M], where M is the number of subspaces
        Returns:
            feature: [n, D], where D = M * dsub, and dsub is the dimension of subspace

        """
        n, MM = codes.shape
        cen = self.centroids_torch   # [M, ksub, dsub]
        M, ksub, dsub = cen.shape
        assert MM == M, f"input codes have {MM} subspace, but quantizer have {M} subspace"

        # for loop version
        # x = torch.empty((n, M * dsub), dtype=torch.float32, device=codes.device)
        # for m in range(M):
        #     xslice = cen[m, codes[:, m].long()]
        #     x[:, m * dsub:(m + 1) * dsub] = xslice

        # parallel version
        # x[n, m, j] = cen[m][codes[n][m]][j]
        x = torch.gather(
            cen.unsqueeze(0).expand(n, -1, -1, -1),  # [n, M, ksub, dsub]
            dim=2,
            index=codes.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, dsub)  # [n, M, 1, dsub]
        )   # [n, M, 1, dsub]
        x = x.view(n, -1)

        if self.pre is not None:
            A, b = self.A, self.b
            if b.numel() > 0:
                x -= b
            x = x @ A
        return x


if __name__ == '__main__':
    from tqdm import tqdm
    # codec = faiss.read_index("/data/yuxian/datasets/multi_domain_paper/it/bpe/de-en-bin/quantizer-decoder.new")
    codec = faiss.read_index("/data/yuxian/datasets/multi_domain_paper/law/bpe/de-en-bin/quantizer-decoder.new")

    # test index
    def test_codec(codec, gpu=False):
        x = torch.rand(1000, 1024).numpy()
        if gpu:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()

            # here we are using a 64-byte PQ, so we must set the lookup tables to
            # 16 bit float (this is due to the limited temporary memory).
            co.useFloat16 = True
            codec = faiss.index_cpu_to_gpu(res, 0, codec, co)
        for _ in tqdm(range(1000)):
            codec.sa_encode(x)
    test_codec(codec, True)


    # test encode
    dev = torch.device('cuda:3')
    tcodec = TorchPQCodec(codec).to(dev)

    xb = torch.rand(1000, 1024).to(dev)
    (tcodec.encode(xb).cpu().numpy() == codec.sa_encode(xb.cpu().numpy())).all()
    # test decode
    codes = torch.randint(256, size=(1000, 128), dtype=torch.uint8).to(dev)
    np.allclose(
        tcodec.decode(codes).cpu().numpy(),
        codec.sa_decode(codes.cpu().numpy()),
        atol=1e-6
    )

    # for _ in tqdm(range(100)):
    #     tcodec.encode(xb)
    #
    # for _ in tqdm(range(100)):
    #     tcodec.decode(codes)

    src = torch.rand(5, 1024).to(dev)
    tgt = torch.rand(200000, 1024).to(dev)
    src_code = torch.randint(256, size=(5, 128), dtype=torch.long).to(dev)
    tgt_code = torch.randint(256, size=(200000, 128), dtype=torch.long).to(dev)

    for _ in tqdm(range(100)):
        tcodec.compute_sim(src_code, tgt_code)
    for _ in tqdm(range(100)):
        torch.matmul(src, tgt.transpose(1, 0))
