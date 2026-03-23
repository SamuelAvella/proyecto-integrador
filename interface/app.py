import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import gradio as gr
from models.integrated_system import IntegratedSystem

system = IntegratedSystem()

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

ICONS = {
    'airplane': '✈️', 'automobile': '🚗', 'bird': '🐦', 'cat': '🐱',
    'deer': '🦌', 'dog': '🐶', 'frog': '🐸', 'horse': '🐴',
    'ship': '🚢', 'truck': '🚛'
}


def upscale_pixel_art(img, scale=10):
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.NEAREST)

def process(category):
    if not category:
        return None, "Selecciona una categoría", "—"

    image, description, cnn_result = system.run(category)

    # 🔥 Escalar imagen manteniendo píxeles
    image = upscale_pixel_art(image, scale=10)

    # Extraer clase CNN
    cnn_class = cnn_result.split(" (")[0]

    from models.integrated_system import generate_description
    description = generate_description(
        system.lstm, system.tokenizer, system.max_length, cnn_class
    )

    return image, cnn_result, description

css = """
* { box-sizing: border-box; }

body, .gradio-container {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'Segoe UI', sans-serif !important;
}

.main-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, #1e3a5f, #1e40af);
    border-radius: 16px;
    border: 1px solid #2563eb;
}

.header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 0.5rem;
}

.header p {
    color: #93c5fd !important;
    font-size: 1rem;
}

.flow-steps {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.flow-step {
    background: rgba(37, 99, 235, 0.3);
    border: 1px solid #3b82f6;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 0.85rem;
    color: #bfdbfe;
}

.flow-arrow {
    color: #3b82f6;
    font-size: 1.2rem;
}

.section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #93c5fd !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}

.category-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}

.cat-btn {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 0.6rem 0.4rem !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-align: center !important;
}

.cat-btn:hover {
    background: #1e40af !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

.cat-btn.selected {
    background: #1d4ed8 !important;
    border-color: #60a5fa !important;
    color: #ffffff !important;
}

.results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}

.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
}

.image-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

.generate-btn {
    width: 100%;
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.8rem !important;
    cursor: pointer !important;
    margin-bottom: 1.5rem !important;
}

.generate-btn:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
}

label, .label-wrap {
    color: #93c5fd !important;
}

textarea, input {
    background: #0f172a !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

.footer-text {
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e293b;
}
"""

with gr.Blocks(css=css, title="Sistema Integrado CIFAR-10") as demo:
    selected_category = gr.State(value="cat")

    with gr.Column(elem_classes="main-wrapper"):

        # Header
        gr.HTML("""
        <div class="header">
            <h1>🧠 Sistema Integrado CIFAR-10</h1>
            <p>Generación, clasificación y descripción de imágenes mediante redes neuronales</p>
            <div class="flow-steps">
                <span class="flow-step">🎯 Selección</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">🎨 GAN genera imagen</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">🔍 CNN clasifica</span>
                <span class="flow-arrow">→</span>
                <span class="flow-step">📝 LSTM describe</span>
            </div>
        </div>
        """)

        # Selector de categorías
        gr.HTML('<div class="section-title">Selecciona una categoría</div>')

        with gr.Row():
            cat_buttons = []
            for i, cat in enumerate(CIFAR10_CLASSES):
                btn = gr.Button(
                    f"{ICONS[cat]} {cat}",
                    elem_classes="cat-btn",
                    size="sm"
                )
                cat_buttons.append((cat, btn))

        # Categoría seleccionada (display)
        selected_display = gr.Textbox(
            label="Categoría seleccionada",
            value="cat",
            interactive=False
        )

        # Botón generar
        generate_btn = gr.Button("⚡ Generar", elem_classes="generate-btn")

        # Imagen generada
        gr.HTML('<div class="section-title">Imagen generada (GAN)</div>')
        image_output = gr.Image(
            label="",
            type="pil",
            width=300,
            height=300,
            show_label=False
        )

        # Resultados CNN y LSTM
        gr.HTML('<div class="section-title">Resultados</div>')
        with gr.Row():
            cnn_output = gr.Textbox(
                label="🔍 Clasificación CNN",
                lines=2,
                interactive=False
            )
            description_output = gr.Textbox(
                label="📝 Descripción LSTM",
                lines=2,
                interactive=False
            )

        gr.HTML('<div class="footer-text">CGAN (PyTorch) · CNN (TensorFlow/Keras) · LSTM (TensorFlow/Keras)</div>')

    # Lógica botones de categoría
    for cat, btn in cat_buttons:
        btn.click(
            fn=lambda c=cat: c,
            outputs=selected_display
        ).then(
            fn=lambda c=cat: c,
            outputs=selected_category
        )

    # Botón generar
    generate_btn.click(
        fn=process,
        inputs=selected_category,
        outputs=[image_output, cnn_output, description_output]
    )

if __name__ == "__main__":
    demo.launch()