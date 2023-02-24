from __future__ import annotations

from torch import nn


class ProjectionMLP(nn.Sequential):
    """
    Sequential module of (Linear-BN-ReLU) layers as projection MLP on top of the
    visual backbone. BatchNorm in final layer is not followed by ReLU.
    """

    def __init__(
        self, input_dim: int, fc_dims: int | list[int], final_bn: bool = False
    ):
        super().__init__()
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        num_layers = len(fc_dims)
        for k, output_dim in enumerate(fc_dims):
            self.append(nn.Linear(input_dim, output_dim, bias=False))

            if k != num_layers - 1:
                self.append(nn.BatchNorm1d(output_dim))
                self.append(nn.ReLU(inplace=True))
            elif final_bn:
                self.append(nn.BatchNorm1d(output_dim))

            # Input dims for next layer:
            input_dim = output_dim
