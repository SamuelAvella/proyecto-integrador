import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_classes=10, noise_dim=100, embed_dim=128):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.net = nn.Sequential(
            # noise + embedding -> mapa espacial 4x4
            nn.Linear(noise_dim + embed_dim, 4 * 4 * 512, bias=False),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Refinamiento sin upsample
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        emb = self.label_emb(labels)          # (B, embed_dim)
        x = torch.cat([noise, emb], dim=1)    # (B, noise_dim + embed_dim)
        x = self.net(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=128):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Linear(embed_dim, 32 * 32)  # proyectar al tamaño imagen

        self.conv_blocks = nn.Sequential(
            # (3+1) x 32x32 -> 64 x 16x16
            nn.Conv2d(4, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 16x16 -> 128 x 8x8
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.LayerNorm([128, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            # 128 x 8x8 -> 256 x 4x4
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.LayerNorm([256, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            # 256 x 4x4 -> 512 x 2x2
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.LayerNorm([512, 2, 2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        emb = self.label_emb(labels)                    # (B, embed_dim)
        emb = self.label_proj(emb)                      # (B, 32*32)
        emb = emb.view(emb.size(0), 1, 32, 32)         # (B, 1, 32, 32)
        x = torch.cat([images, emb], dim=1)             # (B, 4, 32, 32)
        x = self.conv_blocks(x)
        return self.classifier(x)