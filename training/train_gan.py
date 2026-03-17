import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

from models.conditional_gan import build_generator, build_discriminator


def train():

    # Cargar datos
    (x_train, y_train), _ = cifar10.load_data()

    x_train = (x_train - 127.5) / 127.5  # [-1,1]

    num_classes = 10
    noise_dim = 100

    generator = build_generator(num_classes, noise_dim)
    discriminator = build_discriminator(num_classes)

    discriminator.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # GAN completo
    noise = tf.keras.Input(shape=(noise_dim,))
    label = tf.keras.Input(shape=(1,))

    generated_image = generator([noise, label])
    discriminator.trainable = False

    validity = discriminator([generated_image, label])

    gan = tf.keras.Model([noise, label], validity)

    gan.compile(optimizer='adam', loss='binary_crossentropy')

    # Entrenamiento
    batch_size = 64
    epochs = 2000

    for epoch in range(epochs):

        # -----------------
        # Train Discriminator
        # -----------------
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        real_labels = y_train[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_labels = np.random.randint(0, num_classes, (batch_size, 1))

        fake_imgs = generator.predict([noise, fake_labels], verbose=0)

        d_loss_real = discriminator.train_on_batch([real_imgs, real_labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_imgs, fake_labels], np.zeros((batch_size, 1)))

        # -----------------
        # Train Generator
        # -----------------
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        sampled_labels = np.random.randint(0, num_classes, (batch_size, 1))

        g_loss = gan.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))

        if epoch % 200 == 0:
            print(f"{epoch} [D loss: {d_loss_real}] [G loss: {g_loss}]")

    generator.save("checkpoints/gan_generator.keras")

if __name__ == "__main__":
    train()