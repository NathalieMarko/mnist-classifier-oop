import torch
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load spaCy for negation detection
nlp = spacy.load("en_core_web_sm")

# Load Animal Dictionary & Mapping
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
SYNONYM_TO_ANIMAL = {synonym: animal for animal, synonyms in ANIMAL_DICT.items() for synonym in synonyms}

def get_negated_animals(sentence):
    """Detect negation and return animals that are negated."""
    doc = nlp(sentence)
    negated_animals = set()  # Ensure this is always a set
    
    for token in doc:
        if token.dep_ == "neg":  # Negation found (e.g., "not")
            for child in token.head.children:
                if child.text.lower() in SYNONYM_TO_ANIMAL:
                    negated_animals.add(SYNONYM_TO_ANIMAL[child.text.lower()])

    return negated_animals  # Always return a set, even if empty


def infer_animals_in_text(text, model_path="./models/ner_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs).logits

    predicted_labels = torch.argmax(outputs, dim=2).tolist()[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0], skip_special_tokens=True)

    detected_animals = [tokenizer.decode([inputs["input_ids"][0][i]]).strip().lower() for i, label in enumerate(predicted_labels) if label == 1]
    mapped_animals = set(SYNONYM_TO_ANIMAL.get(animal, animal) for animal in detected_animals)

    negated_animals = get_negated_animals(text)
    final_animals = list(mapped_animals - negated_animals)

    return final_animals


if __name__ == "__main__":
    # Example usage
    test_text = "I see cats, but I do not see a rooster."
    animals_detected = infer_animals_in_text(test_text, model_path="./models/ner_model")
    print("Extracted animals:", animals_detected)

