import requests
from bs4 import BeautifulSoup
import re
import torch
import spacy
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset

# Load spaCy for negation detection
nlp = spacy.load("en_core_web_sm")

# Animal Dictionary with Synonyms
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

# Reverse Mapping (Synonym → Canonical Animal)
SYNONYM_TO_ANIMAL = {synonym: animal for animal, synonyms in ANIMAL_DICT.items() for synonym in synonyms}

# Wikipedia Scraping
WIKI_URLS = [f"https://en.wikipedia.org/wiki/{animal.capitalize()}" for animal in ANIMAL_DICT.keys()]

def scrape_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.text for p in soup.find_all("p")]
    return paragraphs

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)  # Remove citations like [1]
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

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


def annotate_sentences(sentences, tokenizer):
    dataset = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        labels = [0] * len(tokens)  # Default to "O"

        negated_animals = get_negated_animals(sentence)

        for i, token in enumerate(tokens):
            token_lower = token.replace("##", "").lower()
            if token_lower in SYNONYM_TO_ANIMAL and SYNONYM_TO_ANIMAL[token_lower] not in negated_animals:
                labels[i] = 1  # B-ANIMAL

        dataset.append({"tokens": tokens, "ner_tags": labels})
    return dataset

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128
    )

    word_ids = tokenized_inputs.word_ids()
    labels = [-100 if word_id is None else examples["ner_tags"][word_id] for word_id in word_ids]

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def train_ner_model():
    MODEL_NAME = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Get Wikipedia Data
    animal_sentences = []
    for url in WIKI_URLS:
        animal_sentences.extend(scrape_wikipedia(url))

    clean_animal_sentences = [clean_text(sent) for sent in animal_sentences if len(sent.split()) > 5]

    # Prepare Dataset
    ner_dataset = annotate_sentences(clean_animal_sentences, tokenizer)
    dataset = Dataset.from_list(ner_dataset)

    # Tokenize and Align Labels
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=False, remove_columns=["tokens", "ner_tags"])
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset, eval_dataset = split["train"], split["test"]

    # Train Model
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
    training_args = TrainingArguments(
        output_dir="./models/ner_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    # Train and Save Model
    trainer.train()
    trainer.save_model("./models/ner_model")
    tokenizer.save_pretrained("./models/ner_model")

    print("Model training complete! Saved to ./models/ner_model")

if __name__ == "__main__":
    train_ner_model()
