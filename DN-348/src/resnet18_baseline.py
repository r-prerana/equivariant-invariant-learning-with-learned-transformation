import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18IdentityBaseline(nn.Module):
    """
    Standard ResNet-18 Vehicle Identity Baseline for DN-348.
    
    Architecture:
        Input Image: (B, 3, 256, 256)
             ↓
        ResNet-18 Backbone (ImageNet pretrained)
             ↓
        Global Average Pooling (512-D)
             ↓
        Embedding Layer: Linear(512, 1024)
             ↓
        L2 Normalization (1024-D, Unit Hypersphere S^{1023})
             ↓
        Classification Head: Linear(1024, num_classes=200)
    """

    def __init__(self, num_classes=200, embedding_dim=1024, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.inv_repr_size = 512  # For direct compatibility with eval_map_cmc_dn348.py
        self.inv_emb_size = embedding_dim

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Backbone: conv1 through avgpool + flatten to 512-D
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
            nn.Flatten(1)
        )
        
        # 1024-D embedding layer (projector_inv name gives drop-in compatibility with eval_map_cmc_dn348)
        self.projector_inv = nn.Linear(512, embedding_dim)
        
        # 200-class identity classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_features(self, x):
        """Extracts the 1024-D L2-normalized embedding used for retrieval."""
        feat = self.backbone(x)             # (B, 512)
        emb = self.projector_inv(feat)      # (B, 1024)
        emb_norm = F.normalize(emb, dim=-1) # (B, 1024)
        return emb_norm

    def forward(self, x):
        """Forward pass for identity training."""
        feat = self.backbone(x)             # (B, 512)
        emb = self.projector_inv(feat)      # (B, 1024)
        emb_norm = F.normalize(emb, dim=-1) # (B, 1024)
        logits = self.classifier(emb)       # (B, num_classes)
        return logits, emb_norm
