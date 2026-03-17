import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        
class Generator(nn.Module):
    def __init__(self, num_classes=10, noise_dim=100, embed_dim=128):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.main = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 4 * 4 * 512, bias=False),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Unflatten(1, (512, 4, 4)),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x = torch.cat([z, emb], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=128):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Linear(embed_dim, 32 * 32)

        self.main = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        emb = self.label_emb(labels)
        emb = self.label_proj(emb).view(labels.size(0), 1, 32, 32)
        x = torch.cat([img, emb], dim=1)
        return self.main(x)