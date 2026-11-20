# src/quaternion_transform.py
#
# Geometry-aware learned transformation encoder for SIE.
#
# Replaces the arbitrary 32-D STL-style latent (transformation_encoder.py)
# with a representation explicitly constrained to be a unit quaternion --
# matching exactly what the SIE hypernetwork was designed and tuned for
# (the original ground-truth g is always a unit quaternion, dim 4).
#
# ------------------------------------------------------------------
# Why this fixes what the STL-style latent couldn't
# ------------------------------------------------------------------
# 1. Scale: the output is a normalized 4-vector by construction. There
#    is no "raw norm grows unboundedly across training" failure mode
#    possible -- normalize() makes it structurally impossible, not
#    regularized-against.
# 2. Signal: q_pred is trained with DIRECT supervision against the
#    true relative quaternion (available on 3DIEBench) via a proper
#    geodesic loss. This is a dense, always-correct, per-sample
#    training signal from step 1 -- unlike InfoNCE, which depends on
#    finding decent positive pairs in a batch and provides no signal
#    at all until that happens. The ground-truth label supervises the
#    ENCODER only; the hypernetwork/predictor only ever sees q_pred,
#    never the label itself -- this does not reintroduce the
#    dependency the project is trying to remove, since nothing about
#    the final predictor mechanism requires labels to run.
# 3. Correct group structure: inverse consistency is now the actual
#    quaternion inverse relationship (q_BA ~= conjugate(q_AB), i.e.
#    q_AB (x) q_BA ~= identity), not the additive approximation
#    (T_xy + T_yx ~= 0) used previously, which was never a
#    mathematically valid constraint for a multiplicative group.
#
# ------------------------------------------------------------------
# Known remaining risk: DIRECT supervision (2 above) is what prevents
# the constant-output collapse that self-consistency losses alone
# (inverse/composition/cycle) can trivially satisfy with e.g. a fixed
# q_pred = identity for every input. Do not remove the supervised
# term and rely on inverse consistency alone -- that reintroduces
# exactly the collapse failure mode this module was built to avoid.
# ------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.transformation_encoder as te  # reuse load_mlp


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4) in (x, y, z, w) order (matches transformation_pairing's
    existing quaternion_to_angle, which reads w from index 3).
    Conjugate of a unit quaternion = its inverse: negate the vector
    part, keep the scalar part.
    """
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack([-x, -y, -z, w], dim=-1)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product q1 (x) q2, both (..., 4) in (x, y, z, w) order.
    Not used by the MVP loss below (inverse consistency only needs
    conjugate) -- kept here for composition/cycle consistency if you
    extend this later.
    """
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)


def quaternion_geodesic_loss(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """
    Per-sample geodesic distance (radians) between q_pred and
    q_target, averaged over the batch. abs() on the dot product
    handles the q / -q double-cover ambiguity (both represent the
    same rotation).
    """
    q_pred = F.normalize(q_pred, dim=-1)
    q_target = F.normalize(q_target, dim=-1)
    dot = (q_pred * q_target).sum(dim=-1).abs().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return (2.0 * torch.acos(dot)).mean()


def quaternion_inverse_consistency_loss(
    q_fwd_pred: torch.Tensor, q_rev_pred: torch.Tensor
) -> torch.Tensor:
    """
    Correct (multiplicative) inverse consistency: the reverse-pair
    prediction should equal the conjugate (= inverse, for unit
    quaternions) of the forward-pair prediction. Replaces the
    previous MSE(T_xy + T_yx, 0) additive approximation, which was
    not a valid constraint for a multiplicative group (rotations
    compose, they don't sum to identity).
    """
    return quaternion_geodesic_loss(q_rev_pred, quaternion_conjugate(q_fwd_pred))


class QuaternionEncoder(nn.Module):
    """
    Predicts a unit quaternion (relative rotation) from a pair of
    backbone features. Structurally identical input construction to
    the existing TransformationEncoder ([feat1, feat2, feat2-feat1]),
    but the output head is a small final Linear to 4 raw numbers,
    L2-normalized onto the unit quaternion manifold.
    """

    def __init__(self, repr_size: int, hidden_backbone: str = "1024-512-64"):
        super().__init__()
        # Hidden trunk (everything except the final 4-D head) reuses
        # the same load_mlp builder as the STL-style encoder.
        self.net = te.load_mlp(3 * repr_size, hidden_backbone)
        hidden_dim = int(hidden_backbone.split("-")[-1])
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        assert feat1.shape == feat2.shape
        diff = feat2 - feat1
        x = torch.cat([feat1, feat2, diff], dim=-1)
        h = self.net(x)
        q_raw = self.head(h)
        return F.normalize(q_raw, dim=-1)
