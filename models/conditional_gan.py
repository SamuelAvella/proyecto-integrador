import tensorflow as tf
from tensorflow.keras import layers


def build_generator(num_classes, noise_dim=100):

    noise_input = layers.Input(shape=(noise_dim,))
    label_input = layers.Input(shape=(1,))

    # Embedding de la clase
    label_embedding = layers.Embedding(num_classes, noise_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    # Combinar ruido + clase
    x = layers.Concatenate()([noise_input, label_embedding])

    # Red
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)

    x = layers.Dense(32 * 32 * 3, activation='tanh')(x)
    output = layers.Reshape((32, 32, 3))(x)

    return tf.keras.Model([noise_input, label_input], output)

def build_discriminator(num_classes):

    image_input = layers.Input(shape=(32, 32, 3))
    label_input = layers.Input(shape=(1,))

    # Embedding de la clase
    label_embedding = layers.Embedding(num_classes, 32 * 32 * 3)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape((32, 32, 3))(label_embedding)

    # Combinar imagen + clase
    x = layers.Concatenate()([image_input, label_embedding])

    # Red
    x = layers.Conv2D(64, (3,3), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, (3,3), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model([image_input, label_input], x)