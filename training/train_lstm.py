import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.lstm_generator import build_lstm


def load_data():
    with open("data/descriptions.json") as f:
        data = json.load(f)

    texts = []
    for category, descriptions in data.items():
        for desc in descriptions:
            texts.append(f"<start> {category} {desc} <end>")

    return texts


def tokenize(texts):
    tokenizer = Tokenizer(filters='')  
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return sequences, tokenizer


def pad(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded = pad_sequences(sequences, maxlen=max_length, padding='pre')
    return padded, max_length


def create_dataset(padded, max_length):
    X, y = [], []

    for seq in padded:
        for i in range(1, len(seq)):
            X.append(seq[:i])
            y.append(seq[i])

    X = pad_sequences(X, maxlen=max_length - 1, padding='pre')
    y = np.array(y)

    return X, y


def train():
    texts = load_data()
    sequences, tokenizer = tokenize(texts)
    padded, max_length = pad(sequences)
    X, y = create_dataset(padded, max_length)

    vocab_size = len(tokenizer.word_index) + 1

    print(f"Vocab size: {vocab_size}")
    print(f"Max length: {max_length}")
    print(f"Dataset size: {len(X)}")

    model = build_lstm(vocab_size, X.shape[1])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/lstm_model.keras',
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]

    model.fit(
        X, y,
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    with open("checkpoints/tokenizer.pkl", "wb") as f:
        pickle.dump({
            'tokenizer': tokenizer,
            'max_length': max_length
        }, f)

    print("Entrenamiento completado")


if __name__ == "__main__":
    train()