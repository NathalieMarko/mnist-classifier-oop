from models.mnist_classifier_interface import MnistClassifierInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Input(shape=(28, 28)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train(self):
        (X_train, y_train), _ = mnist.load_data()
        X_train = X_train / 255.0  # Normalize pixel values
        y_train = to_categorical(y_train, 10)  # One-hot encoding
        self.model.fit(X_train, y_train, epochs=5, batch_size=32)
    
    def predict(self, X):
        X = X / 255.0  # Normalize input
        return np.argmax(self.model.predict(X), axis=1)

