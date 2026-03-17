import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.conditional_gan import Generator, Discriminator

os.makedirs("generated_images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {DEVICE}")


# ─────────────────────────────────────────
# Visualización con noise fijo
# ─────────────────────────────────────────
FIXED_NOISE  = torch.randn(16, 100, device=DEVICE)
FIXED_LABELS = torch.arange(16, device=DEVICE) % 10


def save_images(generator, epoch):
    generator.eval()
    with torch.no_grad():
        imgs = generator(FIXED_NOISE, FIXED_LABELS).cpu()

    imgs = (imgs + 1) / 2.0  # [-1,1] -> [0,1]

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.clip(imgs[i].permute(1, 2, 0).numpy(), 0, 1))
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"generated_images/epoch_{epoch:05d}.png", dpi=100)
    plt.close()
    generator.train()


# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
def train():
    # Datos — torchvision descarga CIFAR-10 automáticamente
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],   # -> [-1, 1]
                             [0.5, 0.5, 0.5]),
    ])
    dataset    = datasets.CIFAR10(root="./data", train=True,
                                  download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=True, num_workers=4, pin_memory=True)

    num_classes = 10
    noise_dim   = 100
    epochs      = 50

    G = Generator(num_classes, noise_dim).to(DEVICE)
    D = Discriminator(num_classes).to(DEVICE)

    g_opt = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # LR decay: reduce a la mitad en epoch 2500
    g_sched = torch.optim.lr_scheduler.StepLR(g_opt, step_size=2500, gamma=0.5)
    d_sched = torch.optim.lr_scheduler.StepLR(d_opt, step_size=2500, gamma=0.5)

    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        for real_imgs, real_labels in dataloader:
            real_imgs   = real_imgs.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            batch_size  = real_imgs.size(0)

            # Etiquetas con label smoothing solo en reales
            real_y = torch.ones(batch_size, 1, device=DEVICE) * 0.9
            fake_y = torch.zeros(batch_size, 1, device=DEVICE)

            # ── Discriminador ──────────────────────────────────
            noise       = torch.randn(batch_size, noise_dim, device=DEVICE)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)

            with torch.no_grad():
                fake_imgs = G(noise, fake_labels)

            d_opt.zero_grad()
            d_loss = (
                criterion(D(real_imgs, real_labels), real_y) +
                criterion(D(fake_imgs, fake_labels), fake_y)
            )
            d_loss.backward()
            d_opt.step()

            # ── Generador (2 pasos) ────────────────────────────
            g_loss_accum = 0.0
            for _ in range(2):
                noise         = torch.randn(batch_size, noise_dim, device=DEVICE)
                sampled_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)

                g_opt.zero_grad()
                fake_imgs = G(noise, sampled_labels)
                g_loss    = criterion(D(fake_imgs, sampled_labels),
                                      torch.ones(batch_size, 1, device=DEVICE))
                g_loss.backward()
                g_opt.step()
                g_loss_accum += g_loss.item()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss_accum / 2)

        g_sched.step()
        d_sched.step()

        
        print(
          f"Epoch {epoch:4d} | "
          f"D loss: {np.mean(d_losses):.4f} | "
          f"G loss: {np.mean(g_losses):.4f}"
        )

        if epoch % 10 == 0:
            save_images(G, epoch)
            torch.save(G.state_dict(), f"checkpoints/generator_{epoch:05d}.pt")
            torch.save(D.state_dict(), f"checkpoints/discriminator_{epoch:05d}.pt")

    torch.save(G.state_dict(), "checkpoints/generator_final.pt")
    torch.save(D.state_dict(), "checkpoints/discriminator_final.pt")
    print("Entrenamiento completo.")


if __name__ == "__main__":
    train()