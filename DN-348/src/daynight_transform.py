# src/daynight_transform.py
#
# Learned categorical Day/Night transformation encoder for DN-348.
#
# Methodological guarantee:
# - transition_target supervises ONLY the DayNightEncoder via cross-entropy loss.
# - The hypernetwork/predictor receives probs_fwd, NEVER transition_target.
# - The fact that both quaternion and Day/Night have dimension 4 is coincidental:
#     quaternion = 4D continuous representation on unit sphere
#     Day/Night  = 4-class discrete probability simplex:
#                  0 = DD (Day -> Day)
#                  1 = NN (Night -> Night)
#                  2 = DN (Day -> Night)
#                  3 = ND (Night -> Day)

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mlp(input_dim: int, hidden_spec: str) -> nn.Sequential:
    """Builds an MLP trunk from a 'd1-d2-d3' specification string."""
    dims = [input_dim] + [int(d) for d in hidden_spec.split("-")]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DayNightEncoder(nn.Module):
    """
    Predicts categorical Day/Night transformation probabilities from a pair
    of backbone features.
    Input:  concat([feat1, feat2, feat2 - feat1], dim=-1) -> 3 * repr_size
    Trunk:  Linear + ReLU layers (default: 1024 -> 512 -> 64)
    Head:   Linear(64, 4)
    Output: raw logits [B, 4] and softmax probabilities [B, 4]
    """

    def __init__(self, repr_size: int, hidden_backbone: str = "1024-512-64"):
        super().__init__()
        self.net = load_mlp(3 * repr_size, hidden_backbone)
        hidden_dim = int(hidden_backbone.split("-")[-1])
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor):
        assert feat1.shape == feat2.shape, f"Shape mismatch: {feat1.shape} vs {feat2.shape}"
        diff = feat2 - feat1
        x = torch.cat([feat1, feat2, diff], dim=-1)
        h = self.net(x)
        logits = self.head(h)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


def daynight_transformation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Supervised cross-entropy loss for DayNightEncoder."""
    return F.cross_entropy(logits, target)


def daynight_inverse_consistency_loss(
    probs_fwd: torch.Tensor, probs_rev: torch.Tensor
) -> torch.Tensor:
    """
    Label-free mathematical consistency constraint:
    Forward order: [DD, NN, DN, ND] (indices [0, 1, 2, 3])
    Reverse order: [DD, NN, ND, DN] (indices [0, 1, 3, 2])
    """
    inverse_target = probs_fwd[..., [0, 1, 3, 2]]
    return F.mse_loss(probs_rev, inverse_target)


def daynight_transformation_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes Top-1 classification accuracy for monitoring."""
    preds = logits.argmax(dim=-1)
    return (preds == target).float().mean() * 100.0
