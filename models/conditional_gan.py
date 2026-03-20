import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# ─────────────────────────────────────────────────────────────
# GENERATOR
# Recibe: z (ruido) + label → imagen 3×32×32
#
# Cómo funciona el conditioning:
#   - El label se convierte en un embedding de tamaño embed_dim
#   - Se concatena con z → vector de (noise_dim + embed_dim)
#   - Ese vector se reshape a (512, 4, 4) con una ConvTranspose
#   - Luego upsample progresivo hasta 32×32
# ─────────────────────────────────────────────────────────────     
class Generator(nn.Module):
    def __init__(self, num_classes=10, noise_dim=100, embed_dim=50):
        super().__init__()

        # Embedding: convierte un int (label) en un vector denso
        # num_classes=10 → 10 vectores de tamaño embed_dim en una tabla
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # Proyección inicial: (noise_dim + embed_dim) → tensor espacial
        # Equivale al Linear+Unflatten pero en conv, más estable
        self.init_block = nn.Sequential(
            # input: (noise_dim + embed_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim + embed_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        # Upsample progresivo: 4×4 → 8×8 → 16×16 → 32×32
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()               # salida en [-1, 1], igual que los datos normalizados
        )

    def forward(self, z, labels):
        # z: (B, 100)   labels: (B,)
        emb = self.label_emb(labels)            # (B, embed_dim)
        x = torch.cat([z, emb], dim=1)          # (B, 150)
        x = x.view(x.size(0), -1, 1, 1)        # (B, 150, 1, 1)
        x = self.init_block(x)                  # (B, 512, 4, 4)
        return self.main(x)                     # (B, 3, 32, 32)


# ─────────────────────────────────────────────────────────────
# DISCRIMINATOR
# Recibe: imagen 3×32×32 + label → probabilidad real/falso
#
# Cómo funciona el conditioning:
#   - El label se embede y se proyecta a un mapa (1×32×32)
#   - Se concatena como 4º canal a la imagen → (4×32×32)
#   - La red aprende a correlacionar imagen Y clase juntas
#   - Sin Sigmoid al final → BCEWithLogitsLoss lo hace internamente
# ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=50):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        # Proyecta embedding → mapa espacial 32×32
        # Así el discriminador ve "qué clase espero" en cada píxel
        self.label_proj = nn.Sequential(
            nn.Linear(embed_dim, 32 * 32),
            nn.LeakyReLU(0.2, inplace=True)     # pequeña no-linealidad antes de concatenar
        )


        # Entrada: 4 canales (3 imagen + 1 label map)
        # Downsample progresivo: 32×32 → 16×16 → 8×8 → 4×4 → 1×1
        self.main = nn.Sequential(
            # Primera capa sin BN (estándar en DCGAN)
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),    # LeakyReLU en D, no muere con valores negativos

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # (256, 4, 4) → escalar
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            # SIN Sigmoid — BCEWithLogitsLoss lo incluye numéricamente estable
        )

    def forward(self, img, labels):
        # img: (B, 3, 32, 32)   labels: (B,)
        emb = self.label_emb(labels)                        # (B, embed_dim)
        label_map = self.label_proj(emb)                    # (B, 32*32)
        label_map = label_map.view(labels.size(0), 1, 32, 32)  # (B, 1, 32, 32)
        x = torch.cat([img, label_map], dim=1)              # (B, 4, 32, 32)
        return self.main(x).view(-1, 1)                     # (B, 1)