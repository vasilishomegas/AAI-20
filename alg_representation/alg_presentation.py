from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

def preprocess_data(data):
    return data / 255.0  # range between 0 and 1 where 1 corresponds with 255 and 0 with 0

class TF_NN:
    def __init__(self, input_shape=(28, 28), activation='relu'):
        self.dataset = keras.datasets.mnist
        (self.train_images, self.train_labels), (self.validation_images, self.validation_labels) = self.dataset.load_data()
        print("amount of images: ", len(self.train_labels))
        self.train_images = preprocess_data(self.train_images)
        self.validation_images = preprocess_data(self.validation_images)

        # used the model from the tensorflow keras documentation
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dense(10)
        ])

    def compile_modile(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train_model(self, n=10):
        self.model.fit(self.train_images, self.train_labels, epochs=n)

    def evaluate_model(self):
        _, accuracy = self.model.evaluate(self.validation_images, self.validation_labels, verbose=2)
        return accuracy

def main():
    tf_nn = TF_NN()
    tf_nn.compile_modile()
    tf_nn.train_model()
    accuracy = tf_nn.evaluate_model()
    print("Accuracy of the model is: ", accuracy*100, "%")

if __name__ == '__main__':
    main()
