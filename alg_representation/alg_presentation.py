from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(data):
    preprocessed_data = data
    return preprocessed_data

class TF_NN:
    def __init__(self, input_shape=(28, 28), activation='relu'):
        self.dataset = keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.dataset.load_data()

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(128, activation=activation),
            keras.layers.Dense(10)
        ])

    def compile_modile(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])


print(tf.__version__)

