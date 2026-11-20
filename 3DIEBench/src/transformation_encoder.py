##src/transformation_encoder.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# New module: STL-style label-free transformation encoder for SIE.
#
# ------------------------------------------------------------------
# Architecture
# ------------------------------------------------------------------
#   Image1        Image2
#     |             |
#     v             v
#   Backbone      Backbone
#     |             |
#     +----concat---+
#           |
#           v
#   TransformationEncoder
#           |
#           v
#   TransformationProjector
#           |
#           v
#   Transformation Embedding
#           |
#           v
#        InfoNCE (info_nce_loss)
#
# ------------------------------------------------------------------
# Scope
# ------------------------------------------------------------------
# This module knows only about:
#     feature1, feature2  ->  transformation embedding
#
# It must NEVER depend on, or be aware of:
#     dataset identity, 3DIEBench, VeRi, quaternions, rotation
#     angles, vehicle IDs, camera IDs, or any other label.
#
# This is what makes it reusable across SIE, SIE+STL on 3DIEBench,
# and SIE+STL on VeRi (or any future dataset) without modification.
# Any logic that needs the ground-truth transformation z (e.g. to
# decide which pairs are positives for the InfoNCE loss) belongs in
# src/transformation_pairing.py, not here.
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mlp(input_dim: int, layers_str: str) -> nn.Sequential:
    """Generic MLP builder, matching SIE's existing Projector style."""
    sizes = [input_dim] + [int(s) for s in layers_str.split("-")]
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
    return nn.Sequential(*layers)


class TransformationEncoder(nn.Module):
    """
    Learns a latent transformation representation from a pair of backbone
    features.

    Input:
        feat1, feat2:
            Backbone features extracted from two related images.

    Output:
        z_repr:
            Latent transformation representation.

    Purpose:
        During inference, this learned representation replaces the explicit
        transformation input (e.g. quaternion) previously provided to the
        predictor. The encoder itself is completely label-agnostic.
    """
    def __init__(self, repr_size: int, trans_backbone: str):
        super().__init__()
        self.net = load_mlp(3 * repr_size, trans_backbone)
        self.out_dim = int(trans_backbone.split("-")[-1])

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:

        assert feat1.shape == feat2.shape

        diff = feat2 - feat1

        x = torch.cat(
            [
                feat1,
                feat2,
                diff,
            ],
            dim=-1,
        )

        return self.net(x)


class TransformationProjector(nn.Module):
    """
    Projects the transformation encoder's output into the embedding
    space used for the InfoNCE loss. Mirrors SIE's existing
    invariant/equivariant Projector pattern.

    Prevents the raw encoder representation from collapsing and
    keeps it more transferable, exactly as the projector heads do
    for the invariant/equivariant branches in SIE.
    """
    def __init__(self, trans_dim: int, trans_projector: str):
        super().__init__()
        self.net = load_mlp(trans_dim, trans_projector)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TransformationModule(nn.Module):
    """
    Convenience wrapper bundling TransformationEncoder +
    TransformationProjector into a single call, so models.py only
    needs one line instead of managing both modules separately.

    Returns:
        z_repr : encoder output (fed to the hypernetwork predictor)
        z_proj : projected embedding (fed to info_nce_loss)
    """
    def __init__(self, repr_size: int, trans_backbone: str, trans_projector: str):
        super().__init__()
        self.encoder = TransformationEncoder(repr_size, trans_backbone)
        self.projector = TransformationProjector(self.encoder.out_dim, trans_projector)
        self.out_dim = self.encoder.out_dim

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor):

        z_repr = self.encoder(feat1, feat2)
        z_proj = self.projector(z_repr)

        # Debug (only occasionally)
        if torch.rand(1).item() < 0.001:
            print("z_repr std :", z_repr.std().item())
            print("z_proj std :", z_proj.std().item())

        return z_repr, F.normalize(z_proj, dim=1)

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float=0.2) -> torch.Tensor:
    """
    Standard InfoNCE loss between two sets of positive-paired
    embeddings. `temperature` should be supplied from args
    (e.g. args.trans_temp) at the call site; the default here is
    only a fallback.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    n = z1.size(0)
    scores = torch.matmul(z, z.t()) / temperature
    labels = torch.cat([
        torch.arange(n, 2 * n, device=z.device),
        torch.arange(0, n, device=z.device),
    ])
    diag_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    scores = scores.masked_fill(diag_mask, float("-inf"))
    
    if not hasattr(info_nce_loss, "counter"):
        info_nce_loss.counter = 0

    info_nce_loss.counter += 1

    if info_nce_loss.counter % 100 == 0:
        sim = torch.matmul(z1, z2.T)

    return F.cross_entropy(scores, labels)
  
  

