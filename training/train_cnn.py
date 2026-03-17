import tensorflow as tf
from models.cnn_classifier import build_cnn

def load_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    return x_train, y_train, x_test, y_test

def train_cnn():
    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Build the CNN model
    model = build_cnn()

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train the model
    model.fit(
        x_train, y_train,
        epochs=5,  # Just for local demonstration
        batch_size=64,
        validation_data=(x_test, y_test)
    )

    model.save('checkpoints/cnn_model.keras')


if __name__ == "__main__":
    train_cnn()