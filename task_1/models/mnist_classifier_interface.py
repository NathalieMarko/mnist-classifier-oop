from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
