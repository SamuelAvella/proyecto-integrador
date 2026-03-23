import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_lstm():
    model = tf.keras.models.load_model("checkpoints/lstm_model.keras")

    with open("checkpoints/tokenizer.pkl", "rb") as f:
        data = pickle.load(f)

    return model, data['tokenizer'], data['max_length']


def generate(model, tokenizer, max_length, category, num_words=15, temperature=0.7):
    text = f"<start> {category}"

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]

        seq = tf.keras.preprocessing.sequence.pad_sequences(
            [seq],
            maxlen=max_length - 1,
            padding='pre'
        )

        pred = model.predict(seq, verbose=0)[0]

        # temperature sampling
        pred = np.log(pred + 1e-8) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))

        next_idx = np.random.choice(len(pred), p=pred)
        next_word = tokenizer.index_word.get(next_idx, '')

        if next_word in ['', '<start>']:
            continue

        if next_word == '<end>':
            break

        text += ' ' + next_word

    # limpiar salida
    result = text.replace('<start>', '').replace('<end>', '').strip()

    # quitar la categoría inicial
    return result[len(category):].strip()


def test_all(model, tokenizer, max_length, temperatures, repetitions=3):
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"  TEMPERATURA: {temp}")
        print(f"{'='*60}")

        for cat in CIFAR10_CLASSES:
            print(f"\n  [{cat.upper()}]")
            for _ in range(repetitions):
                desc = generate(model, tokenizer, max_length, cat, temperature=temp)
                print(f"    → {desc}")


if __name__ == "__main__":
    print("Cargando LSTM...")
    model, tokenizer, max_length = load_lstm()

    print(f"Vocab: {len(tokenizer.word_index)} | Max length: {max_length}")

    test_all(
        model,
        tokenizer,
        max_length,
        temperatures=[0.5, 0.7, 1.0],
        repetitions=3
    )