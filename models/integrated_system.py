import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class IntegratedSystem:

    def __init__(self):
        self.lstm_model = tf.keras.models.load_model("checkpoints/lstm_model.keras")

        with open("checkpoints/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

    def generate_description(self, category, num_words=5):

        text = category

        for _ in range(num_words):

            sequence = self.tokenizer.texts_to_sequences([text])[0]
            sequence = pad_sequences([sequence], maxlen=self.lstm_model.input_shape[1], padding='pre')

            prediction = self.lstm_model.predict(sequence, verbose=0)

            predicted_index = np.argmax(prediction)

            next_word = ""

            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    next_word = word
                    break

            text += " " + next_word

        return text