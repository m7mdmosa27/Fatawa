# Fatawa

# Question Answering System with Arabic Language

## Introduction
This is a question answering system for the Arabic language. The system takes a user's question as input and returns the most relevant answer from a dataset of pre-defined questions and answers.

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - pyarabic
  - nltk
  - gensim
  - scikit-learn
  - tensorflow
  - streamlit

## Installation
1. Clone the repository:
git clone

2. Install the required libraries:
pip install -r requirements.txt
## Usage
1. Preprocess the dataset:
- Clean the text data
- Tokenize the text data
- Remove stopwords
- Train Word2Vec models (CBOW and Skip Gram)

2. Train the question answering model:
- Tokenize input questions and target answers
- Pad sequences
- Define and train the model using TensorFlow

3. Use the question answering system:
- Run the Streamlit app:
  ```
  streamlit run app.py
  ```
- Enter a question in the text input.
- Choose the Word2Vec model (TFIDF, CBOW, or Skip Gram).
- Select the NLP method or Generative model.

## File Structure
- `data_modified.csv`: Dataset file containing questions and answers.
- `scbow_model.model`: Word2Vec CBOW model.
- `skipgram_model.model`: Word2Vec Skip Gram model.

## Code
### Cleaning Text
To clean the text data, the following steps are performed:
- Remove sequences of the form "[الانسان :20]".
- Remove URLs, email addresses, and special characters.
- Remove numbers.
- Tokenize the text.
- Remove diacritics.
- Remove stopwords.
- Join the cleaned text.

