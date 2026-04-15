import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes=7, latent_dim=100, image_channels=3):
        super(ConditionalGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.5), # Evita el colapso de modo introduciendo ruido estructural
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenamos el ruido z con el embedding de la clase c
        c = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        z = noise.view(noise.size(0), noise.size(1), 1, 1)
        x = torch.cat([z, c], 1)
        return self.net(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=7, image_channels=3):
        super(ConditionalDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 50)
        self.label_projector = nn.Linear(50, 1 * 128 * 128)
        
        def sn_conv(in_c, out_c, kernel, stride, padding):
            # Spectral Norm estabiliza el entrenamiento limitando la fuerza del Discriminador
            return spectral_norm(nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False))

        self.net = nn.Sequential(
            sn_conv(image_channels + 1, 64, 4, 2, 1), # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(64, 128, 4, 2, 1), # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(128, 256, 4, 2, 1), # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(256, 512, 4, 2, 1), # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(512, 1024, 4, 2, 1), # 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            sn_conv(1024, 1, 4, 1, 0), # 1x1
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        c = self.label_projector(c).view(-1, 1, 128, 128)
        x = torch.cat([img, c], 1)
        return self.net(x).view(-1, 1)