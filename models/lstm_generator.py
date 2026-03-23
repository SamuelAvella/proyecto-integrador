import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(vocab_size, max_length, embedding_dim=64, lstm_units=64):
    model = models.Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            mask_zero=True
        ),
        layers.LSTM(lstm_units, dropout=0.2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    return model