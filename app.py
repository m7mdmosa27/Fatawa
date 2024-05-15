# Python In-built packages
from pathlib import Path

# External packages
import streamlit as st
import settings
import helper
import numpy as np


# Setting page layout
st.set_page_config(
    page_title="FATAWA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("FATAWA")

# Sidebar
st.sidebar.header("ML Model Config")

# Task Type 
source_radio = st.sidebar.radio(
    "Select Task", ['NLP Methods', 'Genrative Method'])


# Create a list of options for the dropdown if 'NLP Methods' or 'Genrative Method'
Models_options_NLP = ['TFIDF', 'CBOW', 'SkipGram']
Models_options_Gn = ['seq2seq']


# Display the selected option
st.sidebar.header('Models Type')

if source_radio == 'NLP Methods':
    # Add a dropdown with text before the selection
    selected_option = st.sidebar.selectbox('Select :', Models_options_NLP)
elif source_radio == 'Genrative Method':
    selected_option = st.sidebar.selectbox('Select :', Models_options_Gn)


st.header(selected_option)

# Create Input Text to Enter the Question
user_input = st.text_input("Enter Your qustion")


if source_radio == 'NLP Methods':
    # Check if the Fiald is Empty 
    if user_input: 
        # Take Uers' Qustion and Back with The Answer
        answer, question = helper.ask_question(user_input, selected_option)
        # Display the input text
        st.header("The anwser is: \n" + answer)
        st.header('The Similaer question is: \n' + question  )