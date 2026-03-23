import tensorflow as tf
from models.cnn_classifier import build_cnn

# Verificar GPU disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Evita que TensorFlow reserve toda la VRAM de golpe
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"Usando GPU: {gpus[0]}")
else:
    print("GPU no encontrada, usando CPU")

def load_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    return x_train, y_train, x_test, y_test

def train_cnn():
    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Data augmentation — genera variaciones de las imágenes en cada epoch
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=15,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    # Build the CNN model
    model = build_cnn()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        # Reduce lr si no mejora en 5 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        # Para el entrenamiento si no mejora en 15 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1
        ),
        # Guarda el mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/cnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True, verbose=1
        )
    ]

    model.summary()

    # Train the model
    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=80,  # Just for local demonstration
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    model.save('checkpoints/cnn_model.keras')


if __name__ == "__main__":
    train_cnn()