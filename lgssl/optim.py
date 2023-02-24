from __future__ import annotations

import torch


def set_weight_decay_per_param(
    model: torch.nn.Module,
    weight_decay: float,
    gain_bias_decay: float | None = None,
    exclude_params: list[str] = [],
) -> list[dict]:
    """
    Set weight decay for trainable parameters of a model. This function allows
    setting different weight decay for normalization layers from rest of the
    model. The output param groups can be used to instantiate an optimizer.

    This function is adapted from the Torchvision ImageNet training script.

    Args:
        model: PyTorch module with trainable parameters.
        weight_decay: Weight decay for all params except normalization layers.
        gain_bias_decay: Weight decay for normalization layers and bias parameters
            everywhere in the model. If ``None``, it defaults to ``weight_decay``.
        exclude_params: List of parameter names whose weight decay should be zero.
            For example, this could be learnable softmax temperature parameter.
    """
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    gain_bias_decay = gain_bias_decay or weight_decay
    params = {"regular": [], "gain_bias": [], "excluded": []}
    params_weight_decay = {
        "regular": weight_decay,
        "gain_bias": gain_bias_decay,
        "excluded": 0.0,
    }

    # Hold references to parameters (tensors) in this set to avoid adding
    # duplicates, because some modules have shared weights (word embeddings)
    # and they may get counted twice -- PyTorch does not like it.
    already_added_parameters = set()

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or p in already_added_parameters:
                continue

            # Record current parameter as "visited".
            already_added_parameters.add(p)

            if any([exclude_name in name for exclude_name in exclude_params]):
                # Check the exclude substrings in parameter name.
                params["excluded"].append(p)
            elif isinstance(module, norm_classes) or "bias" in name:
                # Check the module type or `bias` in parameter name, this matching
                # is sufficient for ResNet-like and Transformer modules of PyTorch.
                params["gain_bias"].append(p)
            else:
                params["regular"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups


def get_linear_scaled_lr(base_lr: float, batch_size: int, base_batch_size: int):
    """
    Determine the learning rate to train a model with given ``batch_size`` if
    the learning rate for ``base_batch_size`` is ``base_lr``. Apply the linear
    scaling rule (https://arxiv.org/abs/1706.02677):

        current_lr = base_lr * batch_size / base_batch_size

    Args:
        base_lr: Learning rate when batch size is ``base_batch_size``.
        batch_size: Current batch size used to train a model.
        base_batch_size: Batch size when learning rate is ``base_lr``.

    Returns:
        Learning rate as per the linear scaling rule.
    """
    return base_lr * batch_size / base_batch_size
