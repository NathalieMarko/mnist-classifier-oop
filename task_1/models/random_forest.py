from models.mnist_classifier_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.datasets import mnist

class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
    
    def train(self):
        (X_train, y_train), _ = mnist.load_data()
        X_train = X_train.reshape(len(X_train), -1)  # Flattening images
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        X = X.reshape(len(X), -1)  # Flatten before predicting
        return self.model.predict(X)
