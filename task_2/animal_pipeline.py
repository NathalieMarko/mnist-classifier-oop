# Import the NER inference function to extract animal names from text
from infer_ner import infer_animals_in_text

# Import the image classification function to detect animals in images
from infer_image_classification import infer_image_classification

def check_animal_in_text_and_image(text, image_path):
    """
    Determines whether the animal described in the text matches the animal detected in the image.

    Process:
    1. Extracts animal names from the given text using a Named Entity Recognition (NER) model.
    2. Classifies the animal present in the image using an image classification model.
    3. Compares the results and returns True if the detected animal in the image matches 
       the extracted animal name(s) from the text. Otherwise, returns False.

    Args:
        text (str): The user's input text containing an animal description.
        image_path (str): Path to the image file containing an animal.

    Returns:
        bool: True if the text and image contain the same animal, False otherwise.
    """

    # Step 1: Extract animal names from the text using the NER model
    extracted_animals = infer_animals_in_text(text, model_path="./models/ner_model")

    # Step 2: Predict the animal in the image using the trained image classification model
    predicted_animal = infer_image_classification(
        image_path, 
        model_path="./models/image_classifier/best_model.pth",
        num_classes=10  # Ensure consistency with the number of trained animal classes
    )

    # Convert both extracted animals and predicted animal to lowercase for case-insensitive comparison
    extracted_animals_lower = extracted_animals  # The extracted list is already lowercase
    predicted_lower = predicted_animal  # The predicted class label is already lowercase

    # Step 3: Compare extracted animals with the predicted animal
    return predicted_lower in extracted_animals_lower  # Return True if a match is found

if __name__ == "__main__":
    # Example usage
    user_text = "I think there is a cow in the photo"  # User's text input
    image_path = "test_image.png"  # Path to the test image

    # Run the validation function
    decision = check_animal_in_text_and_image(user_text, image_path)

    # Print the final decision (True if the text matches the image, False otherwise)
    print("Decision:", decision)  # Expected output: True or False
