This code implements multiple classifiers for the MNIST handwritten digit classification task, following an object-oriented approach with an abstract interface.

Part 1. The following three classifiers were implemented, each inheriting from the `MnistClassifierInterface`:

1)RandomForestMnistClassifier
- Uses a Random Forest Classifier from `sklearn.ensemble`.
- Flattens 28x28 grayscale images into 1D feature vectors.
- Trains the model using `RandomForestClassifier(n_estimators=100)`.
- Predicts labels for test samples after flattening them.

2) CnnMnistClassifier
- Uses a Convolutional Neural Network (CNN)
- Consists of:
  - `Conv2D` layer (32 filters, 3×3 kernel, ReLU activation).
  - `MaxPooling2D` layer (2×2 pool size).
  - `Flatten()` layer.
  - `Dense` layers (128 units with ReLU, 10 units with softmax).
- Uses the Adam optimizer with `categorical_crossentropy` loss.
- Normalizes input images (`/ 255.0`) and applies one-hot encoding to labels.
- Trains for 5 epochs with a batch size of 32.

3) NeuralNetworkMnistClassifier
- Uses a fully connected (dense) neural network.
- Consists of:
  - `Flatten()` layer.
  - `Dense(128, activation='relu')` layer.
  - `Dense(10, activation='softmax')` layer.
- Uses Adam optimizer with `categorical_crossentropy` loss.
- Normalizes images (`/ 255.0`) and applies one-hot encoding.
- Trains for 5 epochs with batch size 32.

-------------------------

Part 2. 

Interface & Training Pipeline
- `MnistClassifierInterface`: Defines an abstract `train()` and `predict()` method for all classifiers.
- `train_model(algorithm)`: Calls `train()` on a selected classifier.
- `predict_sample(algorithm, X_sample)`: Makes predictions using a chosen model.

------------------------

Training and Prediction Execution
- The CNN classifier (`algorithm = "cnn"`) is used for training.
- The dataset is loaded using `mnist.load_data()`.
- The model is trained on the MNIST training set.
- The trained model makes predictions on 10 test samples.

-----------------------

Visualization of Predictions
- The function `visualize_predictions()`:
  - Displays the first 10 test images.
  - Shows predicted vs. actual labels.
  - Uses Matplotlib to plot grayscale images.

----------------------

Final Outputs
- Training completion message for the CNN model.
- Predictions vs. actual labels printed.
- Visualization of sample predictions.


This implementation successfully trains multiple MNIST classifiers and visualizes predictions using a structured OOP approach!