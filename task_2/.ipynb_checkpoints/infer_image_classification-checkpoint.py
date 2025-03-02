# Loads the trained ResNet50 model and applies it to new images.
# Preprocesses images exactly like in training (resize, normalize).
# Runs inference and returns class label.
# Uses torch.no_grad() to disable gradient calculations (for efficiency).

import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image

# Define paths
MODEL_PATH = "./models/image_classifier/best_model.pth"  # Path to trained model
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']  # Class labels
IMAGE_SIZE = (128, 128)  # Image size used during training

def infer_image_classification(image_path, model_path=MODEL_PATH, num_classes=10):
    """
    Load the saved model and classify a single image.
    Returns the predicted class label.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # 1.  Load the ResNet50 model
    model = resnet50(weights=None) # No need for pretrained weights, we use our own
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Ensure same output layer

    # 2ю  Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # 3. Transform
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) # Same normalization as training
    ])

    # 4. Load image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

    # 5. Perform inference:
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred_idx = torch.max(outputs, 1) # Get class index
    
    pred_idx = pred_idx.item()
    predicted_label = CLASS_NAMES[pred_idx]
                   
    return predicted_label

if __name__ == "__main__":
    # Example usage:
    test_image = "test_image.png"
    predicted_class = infer_image_classification(test_image)
    print("Predicted Class:", predicted_class)
