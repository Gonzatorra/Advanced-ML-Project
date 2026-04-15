import torch
import torch.nn as nn
from torchvision import models

class MultiModalViT(nn.Module):
    def __init__(self, meta_input_dim, num_classes):
        super().__init__()
        # Backbone ViT preentrenado
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.vit.heads.head.in_features
         # Remove the original classification head to get features instead of class scores
        self.vit.heads.head = nn.Identity() 

        # Metadata MLP
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Final classifier that takes both ViT features and metadata features
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        img_features = self.vit(x_img)
        meta_features = self.meta_fc(x_meta)
        combined = torch.cat([img_features, meta_features], dim=1)
        out = self.classifier(combined)
        return out