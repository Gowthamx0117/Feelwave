import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Ensure that necessary NLTK datasets are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset (Replace with your actual dataset)
df = pd.read_csv('emotion_dataset_1000.csv')

# Preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words back into a sentence
    return ' '.join(words)

# Apply preprocessing to the text data
df['Text'] = df['Text'].apply(preprocess_text)

# Train the model
X = df['Text']
y = df['Label']
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# Streamlit user interface
st.title("Emotion Recognition from Text")
st.write("This app predicts the emotion in the given text.")

# Input from the user
user_input = st.text_input("Enter a sentence:")

# Button to get the emotion prediction
if st.button("Predict Emotion"):
    if user_input:
        # Preprocess the user input
        processed_input = preprocess_text(user_input)
        # Transform input into TF-IDF features
        input_tfidf = vectorizer.transform([processed_input])
        # Make prediction
        prediction = model.predict(input_tfidf)[0]
        st.write(f"Predicted Emotion: {prediction}")
    else:
        st.write("Please enter a sentence to predict its emotion.")
