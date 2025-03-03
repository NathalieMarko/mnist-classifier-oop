import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import split_data  # Importing the module responsible for splitting the dataset

def train_image_classifier(
    data_dir='/Users/nataliamarko/Documents/GitHub/mnist-classifier-oop/task_2/data/',
    output_dir="./models/image_classifier",
    num_classes=10,
    batch_size=64,
    epochs=5,
    learning_rate=1e-3,
    val_ratio=0.3
):
    """
    Train a ResNet50 model to classify 10 animal classes using resized images and enhanced data augmentation.
    
    Args:
        data_dir (str): Path to the dataset.
        output_dir (str): Directory to save the trained model.
        num_classes (int): Number of animal classes.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training iterations.
        learning_rate (float): Learning rate for the optimizer.
        val_ratio (float): Fraction of data used for validation.
    """

    # Define the base directory where the train and validation directories will be created
    base_directory = '/Users/nataliamarko/Documents/GitHub/mnist-classifier-oop/task_2/'
    train_dir = os.path.join(base_directory, 'train')
    val_dir = os.path.join(base_directory, 'val')

    # Check if the train and validation directories exist; if not, split the dataset
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Splitting dataset into train/val sets...")
        split_data.split_dataset(source_dir=data_dir, base_dir=base_directory, val_ratio=val_ratio)

    # Define image transformations for training (augmentation + normalization)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128 pixels
        transforms.RandomHorizontalFlip(),  # Apply random horizontal flip (data augmentation)
        transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),  # AutoAugment policy for better generalization
        transforms.ToTensor(),  # Convert images to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize using ImageNet mean & std
    ])

    # Define image transformations for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize using ImageNet statistics
    ])

    # Load the dataset using ImageFolder (expects a folder structure where each class has its own folder)
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    # Create DataLoaders for efficient batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for better generalization
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle needed for validation

    # Load a pre-trained ResNet50 model (transfer learning)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Modify the final fully connected layer to match the number of classes in our dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Select the computing device: use GPU if available, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the selected device

    # Define the loss function (CrossEntropyLoss for multi-class classification)
    criterion = nn.CrossEntropyLoss()

    # Use Adam optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track the best validation accuracy to save the best model
    best_val_acc = 0.0

    # Training loop for multiple epochs
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss, total_correct, total_samples = 0, 0, 0

        # Iterate through the training data in batches
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the selected device

            optimizer.zero_grad()  # Reset gradients before backpropagation
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            # Track training metrics
            total_loss += loss.item() * images.size(0)  # Accumulate total loss
            _, preds = torch.max(outputs, 1)  # Get predicted class index
            total_correct += torch.sum(preds == labels).item()  # Count correct predictions
            total_samples += images.size(0)  # Track total processed images
        
        avg_train_loss = total_loss / total_samples  # Compute average training loss
        train_acc = total_correct / total_samples  # Compute training accuracy

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_correct, val_samples = 0, 0

        with torch.no_grad():  # Disable gradient computation for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_samples += images.size(0)

        val_acc = val_correct / val_samples  # Compute validation accuracy

        # Print training progress for the current epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc  # Update best validation accuracy
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # Create directory if not exists
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))  # Save model weights

    print("Training complete. Best validation accuracy:", best_val_acc)

# Run the training function if the script is executed directly
if __name__ == "__main__":
    train_image_classifier(
        data_dir='/Users/nataliamarko/Documents/GitHub/mnist-classifier-oop/task_2/data/',
        output_dir="./models/image_classifier",
        num_classes=10,
        batch_size=64,
        epochs=5,
        learning_rate=1e-3,
        val_ratio=0.3
    )
