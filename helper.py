# from tensorflow.keras.models import Sequential
import streamlit as st
from utilties.modify import WORD2VEC, TFIDF_Model
import settings




def ask_question(question, model_option='TFIDF'):
    if model_option == 'TFIDF':
        return TFIDF_Model(question).get_answer()
    if model_option == 'CBOW':
        return WORD2VEC(question, model_type='CBOW').get_answer()
    if model_option == 'SkipGram':
        return WORD2VEC(question, model_type='SkipGram').get_answer()
    

