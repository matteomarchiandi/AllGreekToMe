import streamlit as st
import pickle
import pandas as pd

# Import your custom logic from model.py
from model import myMNB, text_preprocess

# --- 1. Page Configuration ---
st.set_page_config(page_title="Language Identifier", page_icon="🌍", layout="centered")

# --- 2. Load Models (Cached for performance) ---
# The @st.cache_resource decorator ensures the models are only loaded into memory once.
@st.cache_resource
def load_models():
    with open('nb_model.pkl', 'rb') as f:
        nb = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vect = pickle.load(f)
    return nb, vect

nb, vect = load_models()

# --- 3. App Header ---
st.title("It's All Greek To Me! 🗣️🌍")
st.markdown("""
Welcome to the Language Identifier! This app uses a custom-built **Multinomial Naive Bayes algorithm** implemented entirely
from scratch and trained on a balanced subset of the **WiLI-2018** dataset, featuring 7,000 short text extracts from Wikipedia.

Languages supported: Albanian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hungarian,
Icelandic, Italian, Latvian, Lithuanian, Macedonian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovene, 
Spanish, Swedish, Turkish.

""")

# --- 4. User Input ---
user_input = st.text_area("Enter a short text and get a prediction of the language:")

# --- 5. Prediction & Visualization ---
if st.button("Go!"):
    if not user_input.strip():
        st.warning("Please enter some text to identify.")
    else:
        # Preprocess the input text using your custom function
        cleaned_text = text_preprocess([user_input])
        
        # Transform the text into a sparse matrix
        vectorized_text = vect.transform(cleaned_text)
        
        # Get the prediction and posterior probabilities
        pred_lang, posteriors = nb.predict_posteriors(vectorized_text[0])
        
        # Extract the probability for the winning language
        confidence_score = posteriors[pred_lang]
        
        # Display the predicted result neutrally but prominently
        st.markdown(f"### Predicted Language: **{pred_lang}**")
        
        # st.divider() # Adds a clean horizontal line to separate the sections
        
        # Display the confidence section
        # st.subheader("Confidence Probability")
        
        # Use st.metric to display the confidence score in a large, bold font
        st.metric(label="Model Confidence", value=f"{confidence_score * 100:.2f}%")
        
        # Display the bar chart
        df_probs = pd.DataFrame(list(posteriors.items()), columns=['Language', 'Probability'])
        df_probs.set_index('Language', inplace=True)
        
        st.bar_chart(df_probs)