import pandas as pd
import joblib
import re
from pyarabic.araby import strip_tashkeel
from pyarabic.araby import tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np


import os

# Get the current directory path
current_directory = os.getcwd()
# Load The Dataset
df_modified = pd.read_csv(os.path.join(current_directory, r'data\data_modified.csv'))

stop_words = set(stopwords.words('arabic'))
stop_words.add('وعلى')
def clean_text_arabic(text):
    # Remove sequences of the form "[number:arabic_word]"
    text = re.sub(r'\[[\u0600-\u06FF]+\s*:*\s*\d*\]', '', text)
    # Find URLs and email addresses in the text
    text = re.sub(r'\b(?:https?|ftp)://\S+|www\.\S+|\S+@\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b/*\S*', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = tokenize(text)
    # Remove diacritics
    words = [strip_tashkeel(word) for word in words]
    # Remove stopwords
    cleaned_text = [word for word in words if word not in stop_words]
    # Join words
    cleaned_text = ' '.join(cleaned_text)
    cleaned_text = cleaned_text.replace('الحمد لله والصلاة والسلام رسول الله وآله وصحبه', "")
    cleaned_text = cleaned_text.replace('الحمد لله والصلاة والسلام رسول الله آله وصحبه', "")
    return cleaned_text




class TFIDF_Model:
    def __init__(self, user_question):
        self.posible_ans, self.similar_ques = self.get_most_similar_question_with_TFIDF(user_question)
    
    def get_answer(self):
        return self.posible_ans, self.similar_ques
    
    def get_most_similar_question_with_TFIDF(self, user_question):
        # Preprocess the user question
        user_question = clean_text_arabic(user_question)
        df = df_modified
        # Preprocess dataset questions
        # Load the vectorizer and the transformed dataset
        tfidf_vectorizer = joblib.load(os.path.join(current_directory,r'data\trained_data\tfidf_vectorizer.pkl'))
        tfidf_X = joblib.load(os.path.join(current_directory,r'data\trained_data\X_tfidf.pkl'))
        # Transform the user question
        user_question_vec = tfidf_vectorizer.transform([user_question])
        
        # Calculate cosine similarity between user question and dataset questions
        similarity_scores = cosine_similarity(user_question_vec, tfidf_X)
        
        # Get the index of the most similar question
        most_similar_index = similarity_scores.argmax()
        # Return the index of the most similar question
        return df['ans'].loc[most_similar_index], df['ques'].loc[most_similar_index]
    



class WORD2VEC:
    def __init__(self, user_question, model_type='CBOW'):
        self.model_type = model_type
        self.model = self.load_model()
        self.user_question = user_question
        # self.data2vec = self.get_data2vec()


    def get_word_vectors(self, sentence, model):
        word_vectors = []
        for word in sentence.split():
            try:
                word_vectors.append(model.wv[word])
            except KeyError:
                # If the word is not in the vocabulary, skip it
                continue
        return word_vectors

    def get_sentence_vector(self, sentence_vectors):
        if len(sentence_vectors) > 0:
            return np.mean(sentence_vectors, axis=0)
        else:
            return np.zeros((150,))
        
    def load_model(self):
        if self.model_type == 'CBOW':
            self.data2vec = joblib.load(os.path.join(current_directory,r'data\trained_data\DatasetVectorizeCBOW.pkl'))
            return Word2Vec.load(os.path.join(current_directory,r"emmbedding-models\Our-CBOW\scbow_model.model"))

        elif self.model_type == 'SkipGram':
            self.data2vec = joblib.load(os.path.join(current_directory,r'data\trained_data\DatasetVectorizeSGram.pkl'))
            return Word2Vec.load(os.path.join(current_directory,r"emmbedding-models\Our-SkipGram\skipgram_model.model"))
        
    
    def get_sent2vec_ques(self, question):
        user_question = clean_text_arabic(question)
        user_question = self.get_word_vectors(user_question, self.model)
        return self.get_sentence_vector(user_question)
    

    def get_data2vec(self):
        return np.stack([self.get_sent2vec_ques(question) for question in df_modified['ques_cleaned']])

    def get_answer(self):
        cos = cosine_similarity([self.get_sent2vec_ques(self.user_question)], self.data2vec)
        id = cos.argmax()
        return df_modified['ans'].iloc[id], df_modified['ques'].iloc[id]