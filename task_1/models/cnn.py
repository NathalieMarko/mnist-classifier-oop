from models.mnist_classifier_interface import MnistClassifierInterface
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

class CnnMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(32, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train(self):
        (X_train, y_train), _ = mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        y_train = to_categorical(y_train, 10)
        self.model.fit(X_train, y_train, epochs=5, batch_size=32)
    
    def predict(self, X):
        X = X.reshape(-1, 28, 28, 1) / 255.0
        return np.argmax(self.model.predict(X), axis=1)
