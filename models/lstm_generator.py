import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(vocab_size, max_length, embedding_dim=128, lstm_units=64):

    model = models.Sequential()

    # Embedding (convierte números en vectores)
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True))

    # LSTM
    model.add(layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))

    # Salida
    model.add(layers.Dense(vocab_size, activation='softmax'))

    return model