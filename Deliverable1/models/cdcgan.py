import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes=7, latent_dim=100, image_channels=3):
        super().__init__()
        # 1. LABEL EMBEDDING
        # Converts a class index (0-6) into a dense vector of size 'latent_dim'.
        # This is helpful to give the generator a richer representation of the class information
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        self.net = nn.Sequential(
            # 2. INPUT
            # Takes a noise vector of size 'latent_dim' and a class embedding of size 'latent_dim',
            # concatenates them to form a vector of size 'latent_dim * 2', and reshapes it to (latent_dim * 2, 1, 1).
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            
            # 3. UPSAMPLING BLOCKS (More resolution)
            # Each block consists of a ConvTranspose2d layer followed by BatchNorm and ReLU.
            # This increases the spatial dimensions of the feature maps while reducing the number of channels.
            
            # Layer 1: 512 -> 256   (Output: 4x4 -> 8x8)         
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Layer 2: 256 -> 128  (Output: 8x8 -> 16x16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Layer 3: 128 -> 64 (Output: 16x16 -> 32x32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Layer 4: 64 -> 32 (Output: 32x32 -> 64x64)
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 4. OUTPUT LAYER
            # The final layer transforms the feature maps into an image with 'image_channels'
            # Output: 64x64 -> 128x128 with 3 channels (RGB)
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 5. INPUT PREPOCESSING
        # Concatenate the noise z with the class embedding c
        c = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        z = noise.view(noise.size(0), noise.size(1), 1, 1)
        x = torch.cat([z, c], 1)
        return self.net(x)





class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=7, image_channels=3):
        super().__init__()
        # 1. LABEL EMBEDDING (Spatial expansion)
        # We create an embedding that matches the image resolution (128x128).
        self.label_emb = nn.Embedding(num_classes, 128 * 128)
        
        self.net = nn.Sequential(
            # 2. INPUT LAYER
            # Receives 'image_channels + 1' (3 for RGB + 1 for the label channel).
            # Output: 128x128 -> 64x64
            nn.Conv2d(image_channels + 1, 32, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            
            # 3. DOWNSAMPLING BLOCKS (Feature extraction)
            # Each block reduces spatial dimensions by half and increases depth.
            # Layer 1: 64x64 -> 32x32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4. OUTPUT LAYER (Final Classification)
            # Compresses the 4x4 map into a single scalar value.
            # Output: 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, labels):
        # 5. INPUT PREPROCESSING
        # Join the image and the label channel along the channel dimension (dim 1)
        c = self.label_emb(labels).view(-1, 1, 128, 128)
        x = torch.cat([img, c], 1)
        return self.net(x).view(-1, 1)