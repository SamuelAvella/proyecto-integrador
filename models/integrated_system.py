import os
import pickle
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import io

from models.conditional_gan import Generator
from models.lstm_generator import build_lstm

# ── Configuración ─────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CIFAR10_CLASSES)}


# ── Carga de modelos ──────────────────────────────────────────────────────────

def load_gan(checkpoint_path="checkpoints/best_model3.pt", noise_dim=100, num_classes=10):
    G = Generator(num_classes=num_classes, noise_dim=noise_dim).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    G.load_state_dict(checkpoint['G_state'])
    G.eval()
    return G


def load_cnn(model_path="checkpoints/cnn_model.keras"):
    return tf.keras.models.load_model(model_path)


def load_lstm(model_path="checkpoints/lstm_model.keras",
              tokenizer_path="checkpoints/tokenizer.pkl"):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        data = pickle.load(f)
    tokenizer = data['tokenizer']
    max_length = data['max_length']
    return model, tokenizer, max_length


# ── Generación de imagen (GAN) ────────────────────────────────────────────────

def generate_image(generator, category: str, noise_dim=100):
    if category not in CLASS_TO_IDX:
        raise ValueError(f"Categoría '{category}' no válida.")

    class_idx = CLASS_TO_IDX[category]
    noise = torch.randn(1, noise_dim, device=DEVICE)
    label = torch.tensor([class_idx], device=DEVICE)

    with torch.no_grad():
        img_tensor = generator(noise, label).squeeze(0).cpu()

    img_tensor = torch.clamp(img_tensor, -1, 1)

    img_tensor = (img_tensor + 1) / 2
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return Image.fromarray(img_np)


# ── Clasificación (CNN) ───────────────────────────────────────────────────────

def classify_image(cnn_model, pil_image):
    img = np.array(pil_image.resize((32, 32))) / 255.0
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    preds = cnn_model.predict(img, verbose=0)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return CIFAR10_CLASSES[class_idx], confidence


# ── Generación de descripción (LSTM) ─────────────────────────────────────────

def generate_description(lstm_model, tokenizer, max_length, category: str,
                         num_words=15, temperature=0.7):

    if category not in CLASS_TO_IDX:
        raise ValueError(f"Categoría '{category}' no válida.")

    text = f"<start> {category}"

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]

        seq = tf.keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_length - 1, padding='pre'
        )

        pred = lstm_model.predict(seq, verbose=0)[0]

        pred = np.log(pred + 1e-8) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))

        next_idx = np.random.choice(len(pred), p=pred)
        next_word = tokenizer.index_word.get(next_idx, '')

        if next_word in ['', '<start>']:
            continue

        if next_word == '<end>':
            break

        text += ' ' + next_word

    result = text.replace('<start>', '').replace('<end>', '').strip()

    return result[len(category):].strip()

# ── Sistema integrado ─────────────────────────────────────────────────────────

class IntegratedSystem:
    def __init__(self):
        print("Cargando modelos...")
        self.generator  = load_gan()
        self.cnn        = load_cnn()
        self.lstm, self.tokenizer, self.max_length = load_lstm()
        print("Modelos cargados.")

    def run(self, category: str):
        if category not in CLASS_TO_IDX:
            return None, "Categoría no válida", "—"

        # 1. Generar imagen
        image = generate_image(self.generator, category)

        # 2. Clasificar imagen generada con la CNN
        predicted_class, confidence = classify_image(self.cnn, image)

        # 3. Generar descripción
        description = generate_description(
            self.lstm, self.tokenizer, self.max_length, category
        )

        cnn_result = f"{predicted_class} ({confidence*100:.1f}%)"
        return image, description, cnn_result