
from infer_ner import infer_animals_in_text
from infer_image_classification import infer_image_classification

def check_animal_in_text_and_image(text, image_path):
    """
    1. Extract animal(s) from the text using NER.
    2. Classify the animal in the image.
    3. Compare the results to decide True/False.
    """
    extracted_animals = infer_animals_in_text(text, model_path="./models/ner_model")
    predicted_animal = infer_image_classification(
        image_path, 
        model_path="./models/image_classifier/best_model.pth",
        num_classes=10
    )
    
    extracted_animals_lower = extracted_animals
    predicted_lower = predicted_animal

    # Return True if there's a match
    return predicted_lower in extracted_animals_lower

if __name__ == "__main__":
    user_text = "I think there is a cow in the photo"
    image_path = "test_image.png"
    decision = check_animal_in_text_and_image(user_text, image_path)
    print("Decision:", decision)  # True or False
