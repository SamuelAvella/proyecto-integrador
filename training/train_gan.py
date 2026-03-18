import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from models.conditional_gan import Generator, Discriminator, weights_init
from tqdm import tqdm

os.makedirs("generated_images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────
# Visualización con noise fijo
# ─────────────────────────────────────────
FIXED_NOISE  = torch.randn(16, 100, device=DEVICE)
FIXED_LABELS = torch.arange(16, device=DEVICE) % 10


def save_images(generator, epoch):
    generator.eval()
    with torch.no_grad():
        imgs = generator(FIXED_NOISE, FIXED_LABELS).cpu()

    imgs = (imgs + 1) / 2

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.clip(imgs[i].permute(1, 2, 0).numpy(), 0, 1))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"generated_images/epoch_{epoch:05d}.png")
    plt.close()
    generator.train()


# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
def train():

    # TensorBoard
    writer = SummaryWriter(f"runs/gan_{time.time()}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    num_classes = 10
    noise_dim = 100
    epochs = 50

    # Models
    G = Generator(num_classes, noise_dim).to(DEVICE)
    D = Discriminator(num_classes).to(DEVICE)

    # 🔥 INIT PESOS
    G.apply(weights_init)
    D.apply(weights_init)

    # Optimizadores
    g_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    global_step = 0

    # TRAIN LOOP
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        loop = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{epochs}",
        dynamic_ncols=True,
        leave=True
    )

        for i, (real_imgs, real_labels) in enumerate(loop):

            real_imgs = real_imgs.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            batch_size = real_imgs.size(0)

            # Labels
            real_y = torch.ones(batch_size, 1, device=DEVICE) * 0.9
            fake_y = torch.zeros(batch_size, 1, device=DEVICE)

            # ───────────────
            # Train D
            # ───────────────
            noise = torch.randn(batch_size, noise_dim, device=DEVICE)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)

            fake_imgs = G(noise, fake_labels)

            d_opt.zero_grad()

            d_loss_real = criterion(D(real_imgs, real_labels), real_y)
            d_loss_fake = criterion(D(fake_imgs.detach(), fake_labels), fake_y)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_opt.step()

            # ───────────────
            # Train G
            # ───────────────
            g_loss_accum = 0.0

            for _ in range(2):
                noise = torch.randn(batch_size, noise_dim, device=DEVICE)
                sampled_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)

                g_opt.zero_grad()

                fake_imgs = G(noise, sampled_labels)
                g_loss = criterion(
                    D(fake_imgs, sampled_labels),
                    torch.ones(batch_size, 1, device=DEVICE)
                )

                g_loss.backward()
                g_opt.step()

                g_loss_accum += g_loss.item()

            g_loss_avg = g_loss_accum / 2

            # ───────────────
            # TensorBoard
            # ───────────────
            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator", g_loss_avg, global_step)

            loop.set_postfix({
                "D_loss": f"{d_loss.item():.3f}",
                "G_loss": f"{g_loss_avg:.3f}"
            })

            global_step += 1

        # ───────────────
        # Guardar imágenes + TensorBoard
        # ───────────────
        if epoch % 5 == 0:
            save_images(G, epoch)

            with torch.no_grad():
                fake_imgs = G(FIXED_NOISE, FIXED_LABELS)

                writer.add_images(
                    "Generated",
                    (fake_imgs + 1) / 2,
                    global_step=epoch
                )

        # Guardar modelo
        if epoch % 10 == 0:
            torch.save(G.state_dict(), f"checkpoints/generator_{epoch}.pt")
            torch.save(D.state_dict(), f"checkpoints/discriminator_{epoch}.pt")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} | ⏱ {epoch_time:.1f}s")

    writer.close()
    print("Entrenamiento completado.")


if __name__ == "__main__":
    print(f"Usando: {DEVICE}")
    train()