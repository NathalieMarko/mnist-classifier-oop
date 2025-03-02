import os
import shutil
import random

def split_dataset(source_dir, base_dir, val_ratio=0.2):
    """
    Splits the data into training and validation directories located one level above the source data directory.
    
    Args:
    source_dir (str): The directory where the source data is stored, containing class subdirectories.
    base_dir (str): The base directory to store the train and val directories.
    val_ratio (float): The fraction of data to be used as validation set.
    """
    # Define the train and val directories one level above the source directory
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    # Ensure the base directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Source directory: {source_dir}")
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")

    # Process each class directory in the source directory
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        
        # Skip any files (non-directories) or hidden directories like .ipynb_checkpoints
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        
        print(f"Processing class: {class_name}")

        # Create train and validation subdirectories for the class
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get all images and shuffle them
        images = [image for image in os.listdir(class_dir) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        print(f"Found {len(images)} images for class {class_name}")
        
        # Split the images according to the validation ratio
        num_val_images = int(len(images) * val_ratio)
        val_images = images[:num_val_images]
        train_images = images[num_val_images:]
        
        print(f"Copying {len(train_images)} to training and {len(val_images)} to validation for class {class_name}")
        
        # Copy images to their respective directories
        for image in train_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(train_class_dir, image))
        for image in val_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(val_class_dir, image))

