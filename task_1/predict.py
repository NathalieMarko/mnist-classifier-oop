def predict_sample(algorithm, X_sample):
    model = MnistClassifier(algorithm)
    prediction = model.predict(X_sample)
    print(f"Prediction for {algorithm} model: {prediction}")
