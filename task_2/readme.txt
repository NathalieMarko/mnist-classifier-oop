TASK 2: Named Entity Recognition + Image Classification Pipeline

This project develops a machine learning pipeline that processes both text and image inputs to determine whether a user’s description of an image is correct. It consists of two separate models: 

1. Named Entity Recognition (NER) Model – Extracts animal names from text.
2. Image Classification Model – Identifies animals in images.

The pipeline integrates these models to return a boolean decision:  
- True if the text description matches the detected animal in the image.  
- False otherwise.

--------------------------------

1. Data Collection & Preprocessing
- The Animal-10 Dataset from Kaggle:
https://www.kaggle.com/datasets/viratkothari/animal10 was selected. It contains 10 animal classes:  
  `butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel`
- The dataset was split into training (80) and validation (20%) sets using the `split_dataset()` function.
- Exploratory Data Analysis (EDA):
  - The dataset’s class distribution was visualized.
  - The image size distribution was analyzed.
  - Corrupted files were detected and removed.

----------------------------

2. Image Classification Model
- Model Architecture:  
  - Pretrained ResNet-50 model (fine-tuned for 10 animal classes).  
  - The fully connected layer was modified to output 10 classes.  
  - Images were resized to 128×128 pixels.
- Data Augmentation & Normalization:
  - `RandomHorizontalFlip`, `AutoAugment(IMAGENET)`, `Normalization` (ImageNet mean & std).
- Training Configuration:
  - Optimizer: Adam (`learning_rate = 1e-3`)
  - Loss: Cross-Entropy
  - Batch size: 64
  - Epochs: 5
- Inference Pipeline:
  - Loads the trained model.
  - Preprocesses the input image.
  - Predicts the animal class.

---------------------------

3. Named Entity Recognition (NER) Model
- Objective: Identify animal names in a given text.
- Data Collection:
  - Scraped Wikipedia pages of the 10 animal classes.
  - Preprocessed text (removing citations, extra spaces).
- NER Model:
  - Based on BERT (bert-base-uncased).
  - Labels each token as **Animal (1) or Not-Animal (0).
  - Handles negation detection (e.g., `"There is no cat"`).
- Training Setup:
  - Uses Hugging Face Transformers (`Trainer` API).
  - Dataset split: 80% train, 20% validation.
  - Fine-tuned on Wikipedia sentences.
- Inference Pipeline:
  - Tokenizes text.
  - Predicts named entities.
  - Maps synonyms to canonical animal names.

-----------------------------

4. Final Pipeline: Text + Image Matching
- Step 1: Extract animal names from the user’s text using the NER model.
- Step 2: Classify the animal in the image using the ResNet-50 classifier.
- Step 3: Compare both outputs:
  - If the detected animal in the image matches the animal(s) extracted from the text → return True.
  - Otherwise → return False.

Example Input/Output*
```
user_text = "I think there is a cow in the photo"
image_path = "test_image.png"
decision = check_animal_in_text_and_image(user_text, image_path)
print("Decision:", decision)  # True or False
```
--------------

Final Results:
- Successfully built an NLP + Computer Vision pipeline.  
- Fine-tuned a ResNet-50 classifier for animal recognition.  
- Developed an NER model to extract animal names from user text.  
- Created a functional end-to-end system that validates text-image consistency.
