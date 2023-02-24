# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.autograd as autograd
import torch.distributed as dist


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors, return_start=False):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    rank_i = get_rank()

    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors, 0

    # get sizes
    sizes = torch.tensor([tensors[0].shape[0]], dtype=torch.int64).to(
        device=tensors[0].device
    )
    sizes = tensor_all_gather(sizes).cpu()
    max_size = sizes.max()

    tensor_list = []
    for tensor in tensors:
        # pad tensor
        padded = torch.empty(max_size, *tensor.shape[1:]).to(tensor)
        padded[: sizes[rank_i]] = tensor

        # gather
        tensor_all = [torch.ones_like(padded) for _ in range(world_size)]
        dist.all_gather(tensor_all, padded, async_op=False)  # performance opt

        # slce them up
        tensor_all = [tensor_all[i][: sizes[i]] for i in range(world_size)]
        tensor_all = torch.cat(tensor_all, dim=0)
        tensor_list.append(tensor_all)

    start_i = 0 if rank_i == 0 else sizes[:rank_i].sum().item()

    return tensor_list, start_i


def tensor_all_gather(tensor):
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensor

    tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_all, tensor, async_op=False)  # performance opt
    tensor_all = torch.cat(tensor_all, dim=0)

    return tensor_all


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    rank_i = get_rank()

    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors, 0

    # get sizes
    sizes = torch.tensor([tensors[0].shape[0]], dtype=torch.int64).to(
        device=tensors[0].device
    )
    sizes = tensor_all_gather(sizes).cpu()
    max_size = sizes.max()

    output_tensor = []
    for tensor in tensors:
        # pad tensor
        padded = torch.empty(max_size, *tensor.shape[1:]).to(tensor)
        padded[: sizes[rank_i]] = tensor

        # gather on padded
        tensor_all = GatherLayer.apply(padded)

        # slce them up
        tensor_all = [tensor_all[i][: sizes[i]] for i in range(world_size)]
        output_tensor.append(torch.cat(tensor_all, dim=0))

    start_i = 0 if rank_i == 0 else sizes[:rank_i].sum().item()
    return output_tensor, start_i
