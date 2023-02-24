import faiss
import faiss.contrib.torch_utils
import torch
from tqdm import tqdm

res = faiss.StandardGpuResources()  # use a single GPU


def faiss_knn(
    dataset, query, k, num_gpus=1, batch_size=512, exclude_self=False, pbar=False
):
    _, ch = dataset.shape
    index_cpu = faiss.IndexFlatL2(ch)

    if num_gpus == 1:
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        gpu_index = faiss.index_cpu_to_all_gpus(index_cpu, co=co)

    # construct index
    gpu_index.add(dataset)

    if exclude_self:
        self_index = torch.arange(dataset.shape[0])

    out_D = []
    out_I = []
    b_range = range(0, query.shape[0], batch_size)
    if pbar:
        b_range = tqdm(b_range)

    for b_i in b_range:
        b_end = min(b_i + batch_size, query.shape[0])
        q_batch = query[b_i:b_end]
        D_i, I_i = gpu_index.search(q_batch, k)

        if exclude_self:
            D_i, I_i = gpu_index.search(q_batch, k + 1)

            self_i = self_index[b_i:b_end, None]
            dif_i = I_i != self_i

            D_i = torch.stack([D_i[i][dif_i[i]][:k] for i in range(len(dif_i))], dim=0)
            I_i = torch.stack([I_i[i][dif_i[i]][:k] for i in range(len(dif_i))], dim=0)
        else:
            D_i, I_i = gpu_index.search(q_batch, k)

        out_D.append(D_i)
        out_I.append(I_i)

    out_D = torch.cat(out_D, dim=0)
    out_I = torch.cat(out_I, dim=0)

    return out_D, out_I


def knn_gather(x, idx):
    feat_dim = x.shape[1]
    K = idx.shape[1]
    idx_expanded = idx[:, :, None].expand(-1, -1, feat_dim)
    x_out = x[:, None, :].expand(-1, K, -1).gather(0, idx_expanded)
    return x_out


def faiss_kmeans(dataset, k, nredo, verbose):
    _, ch = dataset.shape

    kmeans = faiss.Kmeans(
        ch, k, niter=300, spherical=True, verbose=verbose, gpu=True, nredo=nredo
    )
    kmeans.train(dataset)

    D, Ind = kmeans.index.search(dataset, 1)
    return D, Ind
