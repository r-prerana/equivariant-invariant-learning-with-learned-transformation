##src/transformation_pairing.py (new, separate — keeps labels out of the encoder module)
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Training-time pairing logic for the transformation InfoNCE loss.
#
# This file is the ONLY place in the transformation-encoder pipeline
# that is allowed to know about ground-truth transformation labels
# (quaternions on 3DIEBench, discrete bins on VeRi, etc.). It decides
# which samples in a batch should be treated as positive pairs for
# info_nce_loss, since transformation_encoder.py itself must remain
# fully label-agnostic and reusable across datasets.
import torch
def quaternion_to_angle(z: torch.Tensor) -> torch.Tensor:
    """
    Rotation magnitude (radians) from a relative-rotation quaternion
    (x, y, z, w). Used only to bucket pairs for the transformation
    loss -- 3DIEBench-specific caller, not used by VeRi.
    """
    w = z[..., 3].clamp(-1.0, 1.0)
    return 2.0 * torch.acos(w.abs())  # abs() handles sign ambiguity
def bucket_by_magnitude(angles: torch.Tensor, num_buckets: int = 8) -> torch.Tensor:
    edges = torch.linspace(0, torch.pi, num_buckets + 1, device=angles.device)
    return torch.bucketize(angles, edges[1:-1])
def bucket_by_discrete_label(labels: torch.Tensor) -> torch.Tensor:
    """
    Direct pass-through bucketing for datasets that already provide
    discrete transition categories (e.g. VeRi orientation-bin pairs).
    Kept separate from bucket_by_magnitude for clarity of intent.
    """
    return labels
import torch.nn.functional as F


def quaternion_geodesic_distance(z: torch.Tensor) -> torch.Tensor:
    """
    Pairwise geodesic distance (radians) between ALL quaternions in a
    batch, using the full 4D quaternion (axis + magnitude), not just
    the scalar rotation angle from quaternion_to_angle().

    Fixes bucket_by_magnitude()'s core flaw: two rotations with the
    same angle but different axes are NOT the same transformation,
    but bucket_by_magnitude() puts them in the same bucket and
    build_positive_pairs() treats them as positives.
    """
    z = F.normalize(z, dim=-1)
    dot = (z @ z.t()).abs().clamp(-1.0, 1.0)  # |<qi,qj>| handles q/-q double cover
    return 2.0 * torch.acos(dot)


def build_positive_pairs_by_distance(
    y_trans: torch.Tensor,
    z: torch.Tensor,
    max_distance: float = 0.35,
):
    """
    For each sample, its positive is its nearest neighbor in true
    quaternion-geodesic distance, provided that neighbor is within
    max_distance radians. Samples with no sufficiently-close neighbor
    are dropped from InfoNCE for this step rather than force-paired
    with an unrelated transformation.
    """
    device = y_trans.device
    n = z.size(0)
    dist = quaternion_geodesic_distance(z)
    dist.fill_diagonal_(float("inf"))
    nearest_dist, nearest_idx = dist.min(dim=1)
    valid = nearest_dist < max_distance
    if valid.sum() < 2:
        return None, None
    idx1 = torch.arange(n, device=device)[valid]
    idx2 = nearest_idx[valid]
    return y_trans[idx1], y_trans[idx2]


def build_positive_pairs(y_trans: torch.Tensor, bucket_ids: torch.Tensor):
    """
    Pairs samples within each bucket using a cyclic permutation,
    ensuring every sample participates exactly once while avoiding
    self-pairs. Returns (None, None) if no valid pairs exist in
    this batch.
    """
    device = y_trans.device
    idx1_list, idx2_list = [], []
    for b in torch.unique(bucket_ids):
        idxs = (bucket_ids == b).nonzero(as_tuple=True)[0]
        if idxs.numel() < 2:
            continue
        perm = torch.roll(idxs, shifts=1)
        idx1_list.append(idxs)
        idx2_list.append(perm)
    if not idx1_list:
        return None, None
    idx1 = torch.cat(idx1_list)
    idx2 = torch.cat(idx2_list)
    return y_trans[idx1], y_trans[idx2]
