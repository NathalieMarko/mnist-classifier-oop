import requests
from bs4 import BeautifulSoup
import re
import torch
import spacy
from transformers import (AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification)
from datasets import Dataset

# Load spaCy NLP model for negation detection
nlp = spacy.load("en_core_web_sm")

# Define a dictionary of animal names with their synonyms
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

# Generate Wikipedia URLs for each animal
WIKI_URLS = [f"https://en.wikipedia.org/wiki/{animal.capitalize()}" for animal in ANIMAL_DICT.keys()]

def scrape_wikipedia(url):
    """
    Fetches and extracts text paragraphs from a Wikipedia page.

    Args:
        url (str): The Wikipedia page URL.

    Returns:
        list: A list of paragraph texts from the Wikipedia page.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.text for p in soup.find_all("p")]
    return paragraphs

def clean_text(text):
    """
    Cleans extracted text by removing citations and normalizing spaces.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\[\d+\]", "", text)  # Remove references like [1], [2] from Wikipedia
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

def get_negated_animals(sentence):
    """
    Detects negation in a sentence and returns the set of negated animal names.

    Args:
        sentence (str): The input sentence.

    Returns:
        set: A set of negated animal names.
    """
    doc = nlp(sentence)  # Process sentence using spaCy
    negated_animals = set()  # Store detected negated animals
    
    for token in doc:
        if token.dep_ == "neg":  # Look for negation dependency (e.g., "not", "no")
            for child in token.head.children:
                if child.text.lower() in SYNONYM_TO_ANIMAL:
                    negated_animals.add(SYNONYM_TO_ANIMAL[child.text.lower()])  # Map synonym to canonical name

    return negated_animals  # Always return a set, even if empty

def annotate_sentences(sentences, tokenizer):
    """
    Tokenizes sentences and assigns NER labels (1 for animal names, 0 for others).

    Args:
        sentences (list): List of sentences.
        tokenizer (AutoTokenizer): Tokenizer used for tokenizing input text.

    Returns:
        list: List of dictionaries containing tokenized text and corresponding NER tags.
    """
    dataset = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)  # Tokenize the sentence
        labels = [0] * len(tokens)  # Initialize labels as 0 (non-animal)

        negated_animals = get_negated_animals(sentence)  # Detect negated animals

        for i, token in enumerate(tokens):
            token_lower = token.replace("##", "").lower()  # Remove subword prefixes
            if token_lower in SYNONYM_TO_ANIMAL and SYNONYM_TO_ANIMAL[token_lower] not in negated_animals:
                labels[i] = 1  # Assign label 1 to recognized (non-negated) animal names

        dataset.append({"tokens": tokens, "ner_tags": labels})
    return dataset

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes text and aligns labels for Named Entity Recognition (NER) training.

    Args:
        examples (dict): Dictionary containing tokens and labels.
        tokenizer (AutoTokenizer): Tokenizer used for tokenization.

    Returns:
        dict: Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128
    )

    word_ids = tokenized_inputs.word_ids()  # Get word IDs corresponding to tokens
    labels = [-100 if word_id is None else examples["ner_tags"][word_id] for word_id in word_ids]

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_ner_model():
    """
    Trains a Named Entity Recognition (NER) model to detect animal names in text.

    Steps:
    1. Scrape Wikipedia pages for animal-related text.
    2. Clean and preprocess the text data.
    3. Annotate sentences with NER labels (1 for animals, 0 for others).
    4. Tokenize and prepare dataset for training.
    5. Train a transformer-based NER model (BERT).
    6. Save the trained model and tokenizer.
    """
    MODEL_NAME = "bert-base-uncased"  # Pretrained BERT model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Step 1: Scrape Wikipedia pages for animal-related sentences
    animal_sentences = []
    for url in WIKI_URLS:
        animal_sentences.extend(scrape_wikipedia(url))

    # Step 2: Clean the text and filter out short sentences
    clean_animal_sentences = [clean_text(sent) for sent in animal_sentences if len(sent.split()) > 5]

    # Step 3: Annotate sentences with NER labels
    ner_dataset = annotate_sentences(clean_animal_sentences, tokenizer)
    dataset = Dataset.from_list(ner_dataset)  # Convert to Hugging Face dataset format

    # Step 4: Tokenize dataset and align labels
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=False, remove_columns=["tokens", "ner_tags"])
    
    # Step 5: Split dataset into training and validation sets (80% train, 20% validation)
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset, eval_dataset = split["train"], split["test"]

    # Step 6: Load pre-trained BERT model for token classification
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Define training parameters
    training_args = TrainingArguments(
        output_dir="./models/ner_model",  # Directory to save model checkpoints
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at each epoch
        logging_strategy="steps",  # Log every few steps
        logging_steps=10,
        num_train_epochs=3,  # Train for 3 epochs
        per_device_train_batch_size=8,  # Batch size for training
        per_device_eval_batch_size=8,  # Batch size for evaluation
        learning_rate=5e-5,  # Learning rate for optimizer
        logging_dir="./logs",  # Directory for training logs
    )

    # Initialize Trainer API for training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    # Step 7: Train the model
    trainer.train()

    # Step 8: Save the trained model and tokenizer
    trainer.save_model("./models/ner_model")
    tokenizer.save_pretrained("./models/ner_model")

    print("Model training complete! Saved to ./models/ner_model")

# Run the script if executed directly
if __name__ == "__main__":
    train_ner_model()
