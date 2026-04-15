import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalSimpleCNN(nn.Module):
    def __init__(self, input_channels=3, meta_input_dim=10, num_classes=7, dropout_rate=0.5):
        super(MultiModalSimpleCNN, self).__init__()
        
        # --- CNN (Same as CNN Baseline) ---
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # --- Image (features) ---
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(dropout_rate)
        
        # --- Metadata branch ---
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Only change the input dimension to account for the metadata features
        self.fc2 = nn.Linear(256 + 32, num_classes)
    
    def forward(self, x_img, x_meta):
        # --- CNN ---
        x = F.relu(self.bn1(self.conv1(x_img)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC imagen
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # --- Metadata ---
        meta = self.meta_fc(x_meta)
        
        # --- Fusion ---
        x = torch.cat([x, meta], dim=1)
        
        # Final classification layer
        x = self.fc2(x)
        
        return x