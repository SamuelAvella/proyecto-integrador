# Sistema Integrado CIFAR-10: GAN + CNN + LSTM

Sistema de deep learning que integra tres redes neuronales para generar imágenes sintéticas, clasificarlas y describirlas textualmente, usando el dataset CIFAR-10 (10 categorías, 60.000 imágenes 32×32).

---

## Flujo del sistema

```
Categoría seleccionada
        ↓
  GAN genera imagen
        ↓
  CNN clasifica imagen
        ↓
  LSTM describe resultado
```

---

## Estructura del proyecto

```
proyecto-integrador/
├── checkpoints/
│   ├── best_model.pt          # Mejor modelo GAN (por FID)
│   ├── cnn_model.keras        # Modelo CNN entrenado
│   ├── lstm_model.keras       # Modelo LSTM entrenado
│   └── tokenizer.pkl          # Tokenizer + max_length de la LSTM
├── data/
│   └── descriptions.json      # 20 descripciones por clase (200 total)
├── generated_images/          # Imágenes generadas durante el entrenamiento GAN
├── interface/
│   └── app.py                 # Interfaz Gradio
├── models/
│   ├── cnn_classifier.py      # Arquitectura CNN
│   ├── conditional_gan.py     # Arquitectura Generator + Discriminator
│   ├── integrated_system.py   # Sistema integrado (carga y coordina los tres modelos)
│   └── lstm_generator.py      # Arquitectura LSTM
├── training/
│   ├── train_cnn.py           # Entrenamiento CNN
│   ├── train_gan.py           # Entrenamiento GAN
│   └── train_lstm.py          # Entrenamiento LSTM
├── requirements.txt
└── README.md
```

---

## Modelos

### GAN — Conditional GAN (CGAN)

Genera imágenes de 32×32 condicionadas por clase. Arquitectura basada en DCGAN con label embedding.

| Componente | Detalles |
|---|---|
| Framework | PyTorch |
| Generator | 4 capas ConvTranspose2d, 1×1 → 32×32 |
| Discriminator | 3 capas Conv2d + label map como canal extra |
| Noise dim | 100 |
| Embed dim | 50 |
| Loss | BCEWithLogitsLoss + label smoothing (0.9) |
| Optimizer | Adam (lr=0.0002, β=0.5) |
| Scheduler | StepLR (step=75, γ=0.6) |
| Epochs | 300 |
| Métrica | FID (Fréchet Inception Distance) — **mejor: ~38** |

El Discriminator recibe el label como un mapa espacial concatenado como cuarto canal de la imagen. El Generator concatena el ruido con el embedding de clase antes de la primera capa convolucional.

### CNN — Clasificador

Clasifica imágenes CIFAR-10 en 10 categorías con alta precisión.

| Componente | Detalles |
|---|---|
| Framework | TensorFlow / Keras |
| Arquitectura | 3 bloques conv (doble capa) + BN + Dropout + Dense |
| Filtros | 32 → 64 → 128 |
| Regularización | BatchNormalization + Dropout (0.2 / 0.3 / 0.4 / 0.5) |
| Data augmentation | Flip, shift, rotación, zoom |
| Optimizer | Adam (lr=0.001) con ReduceLROnPlateau |
| Epochs | 66 (EarlyStopping en epoch 51 best) |
| **Val accuracy** | **87.79%** |

### LSTM — Generador de descripciones

Genera descripciones textuales dado un nombre de categoría, entrenada con predicción del siguiente token.

| Componente | Detalles |
|---|---|
| Framework | TensorFlow / Keras |
| Arquitectura | Embedding → LSTM × 2 → Dense (softmax) |
| Embedding dim | 128 |
| LSTM units | 128 → 64, Dropout (0.2) |
| Dataset | 20 descripciones × 10 clases = 200 frases |
| Optimizer | Adam (lr=0.001) |
| Epochs | 100 (EarlyStopping patience=10) |

---

## Resultados

| Modelo | Métrica | Resultado |
|---|---|---|
| GAN | FID (↓ mejor) | ~38 en epoch 160 |
| CNN | Val Accuracy | 87.79% |
| LSTM | Loss | Convergencia en ~60 epochs |

---

## Instalación

```bash
git clone <repo>
cd proyecto-integrador
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

---

## Entrenamiento de la GAN (requiere GPU con CUDA)

La GAN usa PyTorch con CUDA. Antes de instalar PyTorch necesitas saber tu versión de CUDA:
```bash
nvidia-smi
```

Busca la línea **CUDA Version** en la esquina superior derecha. Luego visita [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally), selecciona tu versión de CUDA y copia el comando. Ejemplos:
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

--- 

Verifica que PyTorch detecta tu GPU antes de entrenar:
```python
import torch
print(torch.cuda.is_available())     # debe devolver True
print(torch.cuda.get_device_name(0)) # debe mostrar el nombre de tu GPU
```

Si devuelve `False`, la versión de CUDA de PyTorch no coincide con la instalada en tu sistema. Vuelve a pytorch.org y selecciona la versión correcta.
## Uso

### Lanzar la interfaz

```bash
python interface/app.py
```

Abre el navegador en `http://localhost:7860`

### Entrenar los modelos desde cero

```bash
# CNN
python -m training.train_cnn

# LSTM
python -m training.train_lstm

# GAN
python -m training.train_gan
```
## Modelos preentrenados

Los modelos entrenados están disponibles en Google Drive:

[Descargar modelos](https://drive.google.com/drive/folders/1mihc_taEH_LSRt_gOxq6Ru_g-x_28dSx?usp=drive_link)

Descarga los archivos y colócalos en la carpeta `checkpoints/`:
```
checkpoints/
├── best_model.pt
├── cnn_model.keras
└── lstm_model.keras
```
---

## Interfaz

La interfaz Gradio permite seleccionar cualquiera de las 10 categorías CIFAR-10 mediante botones visuales. Al pulsar **Generar**:

1. La **GAN** genera una imagen sintética de esa clase
2. La **CNN** clasifica la imagen generada (con porcentaje de confianza)
3. La **LSTM** genera una descripción textual basada en lo que detectó la CNN

---

## Requisitos

```
torch
torchvision
tensorflow
gradio
pytorch-fid
scipy
numpy
pillow
tqdm
tensorboard
```

---

## Notas técnicas

- La GAN fue entrenada en Google Colab (T4) y localmente (GTX 1660)
- El FID se calcula cada 10 epochs sobre 5.000 muestras usando InceptionV3
- El mejor modelo se guarda automáticamente en `checkpoints/best_model.pt`
- Los checkpoints completos (para reanudar) se guardan cada 20 epochs
