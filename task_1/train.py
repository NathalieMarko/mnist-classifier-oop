def train_model(algorithm):
    model = MnistClassifier(algorithm)
    model.train()
    print(f"Training completed for {algorithm} model.")
