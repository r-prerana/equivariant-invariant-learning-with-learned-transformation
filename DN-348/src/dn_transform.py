import torch
import torch.nn as nn
import torch.nn.functional as F


class DNTransformationEncoder(nn.Module):
    """
    Transformation Encoder for DN-348.

    Predicts the pairwise Day/Night transformation type:

        0 = Day -> Day
        1 = Night -> Night
        2 = Day -> Night
        3 = Night -> Day

    The encoder receives two image embeddings and predicts
    a 4-dimensional transformation representation.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes=(1024, 512, 64),
        num_classes=4,
    ):
        super().__init__()

        layers = []

        # Same pair representation used by the VeRi
        # QuaternionEncoder:
        #
        # [feat1, feat2, feat2 - feat1]
        in_dim = input_size * 3

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        # Four transformation classes
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: [B, input_size]
            feat2: [B, input_size]

        Returns:
            t_pred: [B, 4]

        t_pred is a probability distribution over:

            0 = Day -> Day
            1 = Night -> Night
            2 = Day -> Night
            3 = Night -> Day
        """

        pair_features = torch.cat(
            [
                feat1,
                feat2,
                feat2 - feat1,
            ],
            dim=1,
        )

        logits = self.mlp(pair_features)

        t_pred = F.softmax(logits, dim=1)

        return t_pred


def dn_transformation_loss(t_pred, target):
    """
    Cross-entropy loss for Transformation Encoder supervision.

    Args:
        t_pred:
            [B, 4] predicted transformation probabilities

        target:
            [B] integer transformation labels
            in {0, 1, 2, 3}

    Returns:
        scalar loss
    """

    log_probs = torch.log(
        t_pred.clamp_min(1e-8)
    )

    return F.nll_loss(
        log_probs,
        target,
    )


def dn_transformation_accuracy(t_pred, target):
    """
    Classification accuracy of the Transformation Encoder.
    """

    prediction = t_pred.argmax(dim=1)

    return (
        prediction == target
    ).float().mean()

