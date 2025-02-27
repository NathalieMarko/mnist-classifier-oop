Task 2. Named entity recognition + image classification
In this task, you will work on building your ML pipeline that consists of 2 models responsible for
totally different tasks. The main goal is to understand what the user is asking (NLP) and check if
he is correct or not (Computer Vision).
You will need to:
●
find or collect an animal classification/detection dataset that contains at least 10
classes of animals.
●
train NER model for extracting animal titles from the text. Please use some
transformer-based model (not LLM).
●
Train the animal classification model on your dataset.
●
Build a pipeline that takes as inputs the text message and the image.
In general, the flow should be the following:
1. The user provides a text similar to “There is a cow in the picture.
” and an image that
contains any animal.
2. Your pipeline should decide if it is true or not and provide a boolean value as the output.
You should take care that the text input will not be the same as in the example, and the
user can ask it in a different way.
The solution should contain:
●
Jupyter notebook with exploratory data analysis of your dataset;
●
Parametrized train and inference .py files for the NER model;
●
Parametrized train and inference .py files for the Image Classification model;
●
Python script for the entire pipeline that takes 2 inputs (text and image) and provides
1 boolean value as an output;
