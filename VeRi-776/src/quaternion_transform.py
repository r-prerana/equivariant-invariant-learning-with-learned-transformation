# src/quaternion_transform.py
#
# Learned continuous transformation encoder, validated on 3DIEBench
# (V3-Quat: 72.3% Top-1, R2=0.445 at 200 epochs) and now applied to
# VeRi using the SAME mechanism, pointed at a continuous quaternion
# target constructed from VeRi's discrete orientation annotations via
# DatasetVeRi.orientation_to_quaternion (already validated: the oracle
# run feeding this target directly to the hypernetwork got 64.5% /
# R2=0.49, confirming the target itself is sound before any encoder
# is added).
#
# z (the quaternion built from orientation labels) supervises ONLY
# the encoder below, via a geodesic loss. The hypernetwork/predictor
# receives q_pred, never z -- z is not needed at inference, only
# during training to teach the encoder.

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_mlp(input_dim: int, hidden_spec: str) -> nn.Sequential:
    """
    Builds a plain MLP trunk from a "d1-d2-d3" spec string, ReLU after
    every layer (including the last hidden one -- the final 4-D head
    is a separate, un-activated Linear, added in QuaternionEncoder
    below).
    """
    dims = [input_dim] + [int(d) for d in hidden_spec.split("-")]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4) in (x, y, z, w) order -- matches scipy's
    Rotation.as_quat() convention, which is what
    DatasetVeRi.orientation_to_quaternion already produces.
    Conjugate of a unit quaternion = its inverse: negate the vector
    part, keep the scalar part.
    """
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack([-x, -y, -z, w], dim=-1)


def quaternion_geodesic_loss(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """
    Per-sample geodesic distance (radians) between q_pred and
    q_target, averaged over the batch. abs() on the dot product
    handles the q / -q double-cover ambiguity (both represent the
    same rotation) -- without this, the encoder would be penalized
    for outputting the "other" valid sign, which is not a real error.
    """
    q_pred = F.normalize(q_pred, dim=-1)
    q_target = F.normalize(q_target, dim=-1)
    dot = (q_pred * q_target).sum(dim=-1).abs().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return (2.0 * torch.acos(dot)).mean()


def quaternion_inverse_consistency_loss(
    q_fwd_pred: torch.Tensor, q_rev_pred: torch.Tensor
) -> torch.Tensor:
    """
    Label-free self-consistency: the reverse-pair prediction should
    equal the conjugate (= inverse, for unit quaternions) of the
    forward-pair prediction. Correct multiplicative form -- q_AB (x)
    q_BA ~= identity -- not an additive approximation.
    """
    return quaternion_geodesic_loss(q_rev_pred, quaternion_conjugate(q_fwd_pred))


class QuaternionEncoder(nn.Module):
    """
    Predicts a unit quaternion (relative transformation) from a pair
    of backbone features. Input: [feat1, feat2, feat2-feat1],
    concatenated. Output: 4 raw numbers, L2-normalized onto the unit
    quaternion manifold -- structurally impossible to have unbounded
    scale, by construction, not by regularization.

    This is the SAME architecture validated on 3DIEBench (V3-Quat).
    Nothing dataset-specific here -- what's dataset-specific is only
    the TARGET it's trained against (z), constructed upstream by
    DatasetVeRi.orientation_to_quaternion for VeRi, or the true
    relative rotation for 3DIEBench.
    """

    def __init__(self, repr_size: int, hidden_backbone: str = "1024-512-64"):
        super().__init__()
        self.net = load_mlp(3 * repr_size, hidden_backbone)
        hidden_dim = int(hidden_backbone.split("-")[-1])
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        assert feat1.shape == feat2.shape
        diff = feat2 - feat1
        x = torch.cat([feat1, feat2, diff], dim=-1)
        h = self.net(x)
        q_raw = self.head(h)
        return F.normalize(q_raw, dim=-1)
