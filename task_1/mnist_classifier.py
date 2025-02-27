from models.mnist_classifier_interface import MnistClassifierInterface
from models.random_forest import RandomForestMnistClassifier
from models.neural_network import NeuralNetworkMnistClassifier
from models.cnn import CnnMnistClassifier

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.classifier = RandomForestMnistClassifier()
        elif algorithm == 'nn':
            self.classifier = NeuralNetworkMnistClassifier()
        elif algorithm == 'cnn':
            self.classifier = CnnMnistClassifier()
        else:
            raise ValueError("Unsupported algorithm. Choose from 'rf', 'nn', or 'cnn'.")
    
    def train(self):
        self.classifier.train()
    
    def predict(self, X):
        return self.classifier.predict(X)
