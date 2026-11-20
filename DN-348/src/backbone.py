import sys
from pathlib import Path

import torch
import torch.nn as nn


# Reuse the exact ResNet implementation from the validated VeRi code.
from . import resnet


class DN348Backbone(nn.Module):
    """
    ResNet backbone for the DN-348 experiment.

    This uses the exact ResNet implementation from the validated
    SIE_VeRi_V3Quat experiment.

    For ResNet18:
        input  : [B, 3, 256, 256]
        output : [B, 512]
    """

    def __init__(self, arch="resnet18"):
        super().__init__()

        self.backbone, self.repr_size = resnet.__dict__[arch](
            zero_init_residual=True
        )

    def forward(self, x):
        return self.backbone(x)
