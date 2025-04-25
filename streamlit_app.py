import streamlit as st
import joblib
import os
import numpy as np

# Load model and vectorizer
MODEL_PATH = 'model/genre_classifier.joblib'
VECTORIZER_PATH = 'model/vectorizer.joblib'

@st.cache_resource
def load_model():
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return clf, vectorizer

clf, vectorizer = load_model()

st.title('Movie Genre Classifier')
st.write('Enter a movie name and description to predict its genre.')

movie_name = st.text_input('Movie Name')
description = st.text_area('Movie Description')

if st.button('Predict Genre'):
    if not movie_name or not description:
        st.warning('Please enter both the movie name and description.')
    else:
        text = movie_name + ' ' + description
        vec = vectorizer.transform([text])
        pred = clf.predict(vec)[0]
        proba = clf.predict_proba(vec)[0]
        confidence = np.max(proba)
        st.success(f'Predicted Genre: {pred}')
        st.info(f'Confidence: {confidence:.2%}')
