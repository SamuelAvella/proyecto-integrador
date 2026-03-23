import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend sin GUI, solo guarda archivos
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from models.conditional_gan import Generator, Discriminator, weights_init
from tqdm import tqdm

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from scipy import linalg

os.makedirs("generated_images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIXED_NOISE  = torch.randn(16, 100, device=DEVICE)
FIXED_LABELS = torch.arange(16, device=DEVICE) % 10


# ── FID helpers ───────────────────────────────────────────────────────────────

def get_inception_features(images, inception_model):
    """images: tensor (N, 3, H, W) en [-1, 1] → features (N, 2048)"""
    inception_model.eval()
    images = (images + 1) / 2  # a [0, 1]
    images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        features = inception_model(images)[0]
    return features.squeeze(-1).squeeze(-1).cpu().numpy()


def compute_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_real_features(dataloader, inception_model, max_samples=5000):
    """Calcula features reales una sola vez al inicio del entrenamiento."""
    all_features = []
    collected = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(DEVICE)
        feats = get_inception_features(imgs, inception_model)
        all_features.append(feats)
        collected += len(feats)
        if collected >= max_samples:
            break
    return np.concatenate(all_features, axis=0)[:max_samples]


def evaluate_fid(generator, real_features, inception_model, noise_dim, num_classes, n_samples=5000):
    generator.eval()
    fake_features = []
    batch = 64
    collected = 0
    while collected < n_samples:
        current_batch = min(batch, n_samples - collected)
        noise  = torch.randn(current_batch, noise_dim, device=DEVICE)
        labels = torch.randint(0, num_classes, (current_batch,), device=DEVICE)
        with torch.no_grad():
            imgs = generator(noise, labels)
        feats = get_inception_features(imgs, inception_model)
        fake_features.append(feats)
        collected += current_batch
    fake_features = np.concatenate(fake_features, axis=0)
    generator.train()
    return compute_fid(real_features, fake_features)

# ── Image saving ──────────────────────────────────────────────────────────────

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def save_images(generator, epoch):
    generator.eval()
    with torch.no_grad():
        imgs = generator(FIXED_NOISE, FIXED_LABELS).cpu()
    imgs = (imgs + 1) / 2
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.clip(imgs[i].permute(1, 2, 0).numpy(), 0, 1))
        ax.set_title(CIFAR10_CLASSES[FIXED_LABELS[i].item()], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"generated_images/epoch_{epoch:05d}.png")
    plt.close()
    generator.train()
  
# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(epoch, G, D, g_opt, d_opt, g_scheduler, d_scheduler):
    torch.save({
        'epoch': epoch,
        'G_state': G.state_dict(),
        'D_state': D.state_dict(),
        'g_opt_state': g_opt.state_dict(),
        'd_opt_state': d_opt.state_dict(),
        'g_scheduler_state': g_scheduler.state_dict(),
        'd_scheduler_state': d_scheduler.state_dict(),
    }, f"checkpoints/checkpoint_epoch_{epoch}.pt")


def load_checkpoint(path, G, D, g_opt, d_opt, g_scheduler, d_scheduler):
    checkpoint = torch.load(path, map_location=DEVICE)
    G.load_state_dict(checkpoint['G_state'])
    D.load_state_dict(checkpoint['D_state'])
    g_opt.load_state_dict(checkpoint['g_opt_state'])
    d_opt.load_state_dict(checkpoint['d_opt_state'])
    g_scheduler.load_state_dict(checkpoint['g_scheduler_state'])
    d_scheduler.load_state_dict(checkpoint['d_scheduler_state'])
    return checkpoint['epoch']

# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
def train():

    RESUME_FROM = "checkpoints/checkpoint_epoch_100.pt"
    # RESUME_FROM = "checkpoints/checkpoint_epoch_100.pt"

    writer = SummaryWriter(f"runs/gan_{time.time()}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset   = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    num_classes = 10
    noise_dim   = 100
    epochs      = 300
    
    # Models
    G = Generator(num_classes, noise_dim).to(DEVICE)
    D = Discriminator(num_classes).to(DEVICE)

    # Optimizadores
    g_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    g_scheduler = torch.optim.lr_scheduler.StepLR(g_opt, step_size=75, gamma=0.6)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_opt, step_size=75, gamma=0.6)

    criterion = nn.BCEWithLogitsLoss() 
    
    # ── Inception para FID (se carga una vez) ─────────────────────────────────
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEVICE)
    inception.eval()
    print("Calculando features reales para FID (una sola vez)...")
    real_features = compute_real_features(dataloader, inception, max_samples=5000)
    print(f"Features reales calculadas: {real_features.shape}")

    # ── Cargar checkpoint ─────────────────────────────────────────────────────
    start_epoch = 1
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        start_epoch = load_checkpoint(RESUME_FROM, G, D, g_opt, d_opt, g_scheduler, d_scheduler) + 1
        print(f"Retomando desde epoch {start_epoch}")
    else:
        G.apply(weights_init)
        D.apply(weights_init)

    best_fid = float('inf')
    global_step = 0

    # TRAIN LOOP
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True, leave=True)

        for _, (real_imgs, real_labels) in enumerate(loop):

            real_imgs = real_imgs.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            batch_size = real_imgs.size(0)

            # Labels
            real_y = torch.ones(batch_size, 1, device=DEVICE) * 0.9
            fake_y = torch.zeros(batch_size, 1, device=DEVICE)

            # ── Train D ───────────────────────────────────────────────────────
            noise       = torch.randn(batch_size, noise_dim, device=DEVICE)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
            fake_imgs   = G(noise, fake_labels)

            d_opt.zero_grad()
            d_loss_real = criterion(D(real_imgs, real_labels), real_y)
            d_loss_fake = criterion(D(fake_imgs.detach(), fake_labels), fake_y)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_opt.step()

            for _ in range(1):
                noise          = torch.randn(batch_size, noise_dim, device=DEVICE)
                sampled_labels = torch.randint(0, num_classes, (batch_size,), device=DEVICE)
                g_opt.zero_grad()
                fake_imgs = G(noise, sampled_labels)

                # Le decimos a D que evalúe las imágenes falsas
                # y le pasamos torch.ones porque queremos que D diga "real"
                # — eso significa que G ha engañado al discriminador
                g_loss = criterion(
                    D(fake_imgs, sampled_labels),
                    torch.ones(batch_size, 1, device=DEVICE)
                )
                g_loss.backward()
                g_opt.step()

            # ───────────────
            # TensorBoard
            # ───────────────
            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator", g_loss.item(), global_step)

            loop.set_postfix({
                "D_loss": f"{d_loss.item():.3f}",
                "G_loss": f"{g_loss.item():.3f}",
                "lr_G":   f"{g_scheduler.get_last_lr()[0]:.5f}"
            })

            global_step += 1

        # ── Fin de epoch ──────────────────────────────────────────────────────

        g_scheduler.step()
        d_scheduler.step()

        if epoch % 5 == 0:
            save_images(G, epoch)
            with torch.no_grad():
                writer.add_images("Generated", (G(FIXED_NOISE, FIXED_LABELS) + 1) / 2, global_step=epoch)

        # Guardar modelo
        if epoch % 20 == 0:
            save_checkpoint(epoch, G, D, g_opt, d_opt, g_scheduler, d_scheduler)
        
        # ── FID cada 10 epochs (caro, no hacerlo cada epoch) ─────────────────
        if epoch % 10 == 0:
            fid_score = evaluate_fid(G, real_features, inception, noise_dim, num_classes)
            writer.add_scalar("FID", fid_score, epoch)
            print(f"  FID epoch {epoch}: {fid_score:.2f}")

            if fid_score < best_fid:
                best_fid = fid_score
                torch.save({
                    'epoch': epoch,
                    'G_state': G.state_dict(),
                    'D_state': D.state_dict(),
                    'fid': best_fid,
                }, "checkpoints/best_model.pt")
                print(f"  ✓ Nuevo mejor modelo guardado (FID={best_fid:.2f})")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} | ⏱ {epoch_time:.1f}s | lr_G={g_scheduler.get_last_lr()[0]:.6f}")



    writer.close()
    print("Entrenamiento completado.")


if __name__ == "__main__":
    print(f"Usando: {DEVICE}")
    train()