import torch
import torch.nn as nn
from torchvision import models

class MultiModalEfficientNet(nn.Module):
    def __init__(self, meta_input_dim, num_classes):
        super(MultiModalEfficientNet, self).__init__()
        
        # Pre-trained EfficientNet backbone
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.cnn.classifier[1].in_features
        
        # Remove final layer to get features instead of class scores
        self.cnn.classifier[1] = nn.Identity()  
        
        # Metadata MLP
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier combining CNN + metadata
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x_img, x_meta):
        cnn_features = self.cnn(x_img)
        meta_features = self.meta_fc(x_meta)
        combined = torch.cat([cnn_features, meta_features], dim=1)
        out = self.classifier(combined)
        return out