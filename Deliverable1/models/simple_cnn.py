import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=7, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        # ---- CONVOLUTIONAL LAYERS ---- #
        # Conv layer 1: input_channels --> 32 filters
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # BatchNorm after each conv layer to help with training stability and convergence
        
        # Conv layer 2: 32 filters --> 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv layer 3: 64 filters --> 128 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling layer (2x2) to reduce spatial dimensions after each conv block
        # Useful to get the most important features and reduce computational load
        self.pool = nn.MaxPool2d(2, 2)
        
        
        # ---- FULLY CONNECTED LAYERS ---- #
        # Necessary to get the prediction from the extracted features.
        # The input size is determined by the output of the last conv layer after pooling.
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        
        self.dropout = nn.Dropout(dropout_rate) # Dropout to prevent overfitting,
        # It is inserted after the fully connected layer since  it is the most prone to overfitting,
        # as it has the most parameters and is the last layer before output.
        
        self.fc2 = nn.Linear(256, num_classes)
    
    
    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        
        return x