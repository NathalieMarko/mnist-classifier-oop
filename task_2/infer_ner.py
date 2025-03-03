import torch
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load spaCy for negation detection
nlp = spacy.load("en_core_web_sm")

# Define a dictionary mapping animal names to their synonyms
ANIMAL_DICT = {
    "butterfly": ["butterfly", "butterflies"],
    "cat": ["cat", "cats", "kitty", "kitties", "feline"],
    "chicken": ["chicken", "chickens", "hen", "rooster"],
    "cow": ["cow", "cows", "bull", "calf", "ox"],
    "dog": ["dog", "dogs", "puppy", "puppies", "hound", "canine"],
    "elephant": ["elephant", "elephants", "pachyderm"],
    "horse": ["horse", "horses", "pony", "stallion", "mare"],
    "sheep": ["sheep", "lamb", "ram", "ewe"],
    "spider": ["spider", "tarantula", "arachnid"],
    "squirrel": ["squirrel", "chipmunk"]
}

# Create a reverse mapping from synonyms to their canonical animal name
SYNONYM_TO_ANIMAL = {synonym: animal for animal, synonyms in ANIMAL_DICT.items() for synonym in synonyms}

def get_negated_animals(sentence):
    """
    Detects negation in a given sentence and identifies negated animal names.

    Args:
        sentence (str): Input sentence to analyze.

    Returns:
        set: A set of animal names that are negated in the sentence.
    """
    doc = nlp(sentence)  # Process the sentence using spaCy NLP model
    negated_animals = set()  # Initialize a set to store detected negated animals
    
    for token in doc:
        if token.dep_ == "neg":  # Look for negation dependency (e.g., "not", "no")
            for child in token.head.children:
                if child.text.lower() in SYNONYM_TO_ANIMAL:
                    negated_animals.add(SYNONYM_TO_ANIMAL[child.text.lower()])  # Map synonym to canonical animal name

    return negated_animals  # Return the set of negated animal names (empty if none found)

def infer_animals_in_text(text, model_path="./models/ner_model"):
    """
    Detects animal names in a given text using a pre-trained Named Entity Recognition (NER) model.

    Args:
        text (str): The input text to process.
        model_path (str): Path to the trained NER model.

    Returns:
        list: A list of detected animal names (excluding negated ones).
    """
    # Load the tokenizer and model from the specified path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, return_tensors="pt")

    # Set the model to evaluation mode and disable gradient calculations for inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs).logits  # Get model predictions

    # Convert logits to class labels (0: non-animal, 1: animal)
    predicted_labels = torch.argmax(outputs, dim=2).tolist()[0]

    # Convert token indices back to words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0], skip_special_tokens=True)

    # Extract animal names based on predicted labels
    detected_animals = [
        tokenizer.decode([inputs["input_ids"][0][i]]).strip().lower()
        for i, label in enumerate(predicted_labels) if label == 1
    ]

    # Convert synonyms to their canonical animal names
    mapped_animals = set(SYNONYM_TO_ANIMAL.get(animal, animal) for animal in detected_animals)

    # Detect and exclude negated animal names
    negated_animals = get_negated_animals(text)
    final_animals = list(mapped_animals - negated_animals)  # Remove negated animals

    return final_animals  # Return the final list of detected animals

if __name__ == "__main__":
    # Example usage of the function
    test_text = "I see cats, but I do not a rooster."
    animals_detected = infer_animals_in_text(test_text, model_path="./models/ner_model")
    print("Extracted animals:", animals_detected)  # Output: ['cat'] (since "rooster" is negated)
