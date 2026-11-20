# src/supcon_loss.py
#
# Supervised Contrastive Loss (SupCon) for invariant representations.
# Implements Khosla et al. (NeurIPS 2020) for identity-balanced mini-batches.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss applied to normalized invariant representations.
    
    For each anchor image i in the batch:
      - Positives: other images in the batch sharing the SAME vehicle identity
      - Negatives: images in the batch belonging to DIFFERENT vehicle identities
    
    Gradient routing guarantee:
      Since features are the L2-normalized 1024-D invariant projector output,
      gradients from SupConLoss flow exclusively into:
        invariant projector -> invariant backbone channels -> shared backbone.
      No gradient flows to the Transformation Encoder, Hypernetwork, or equivariant branch.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [N, D], L2-normalized representations.
            labels: Tensor of shape [N], vehicle identity labels (integers).
            domains: Optional Tensor of shape [N], domain labels (0 for Day, 1 for Night).
        Returns:
            loss: Scalar supervised contrastive loss.
        """
        device = features.device
        N = features.shape[0]

        # Dot product similarity matrix scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature

        # Subtract row-max for numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Mask self-contrast
        mask_self = torch.eye(N, dtype=torch.bool, device=device)

        if domains is not None:
            # Cross-Domain-Targeted SupCon (CD-SupCon):
            # Positives: same identity AND opposite domain
            mask_domain_cross = domains.unsqueeze(0) != domains.unsqueeze(1)
            mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & mask_domain_cross

            # Denominator: exclude same-identity same-domain pairs (including self)
            # to prevent false-negative repulsion between images of the same vehicle
            mask_same_id_same_dom = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~mask_domain_cross)
            mask_denom = ~mask_same_id_same_dom
            exp_sim = torch.exp(sim) * mask_denom
        else:
            # Standard SupCon (Baseline)
            mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~mask_self)
            exp_sim = torch.exp(sim) * (~mask_self)

        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        log_prob = sim - log_denom

        pos_counts = mask_pos.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_anchor = -(log_prob * mask_pos).sum(dim=1)[valid] / pos_counts[valid]
        return loss_per_anchor.mean()

