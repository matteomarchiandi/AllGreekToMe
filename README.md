# It's All Greek To Me! 🌍 

## 📌 Overview
This project tackles the task of **Language Identification** for written text. Instead of relying on off-the-shelf machine learning classifiers, I implemented a **Multinomial Naive Bayes (MNB)** algorithm entirely from scratch. 

To make the model accessible to the public, I deployed it as an interactive web application using **Streamlit**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-goes-here.streamlit.app)


## 📊 The Dataset
The model was trained on a balanced subset of the **WiLI-2018 dataset**, consisting of short text extracts from Wikipedia. 
* **Total Samples:** 10,000 documents (7,000 for training, 3,000 for testing)
* **Supported Languages:** Languages supported: Albanian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hungarian, Icelandic, Italian, Latvian, Lithuanian, Macedonian, Maltese, Norwegian, Polish, Portuguese, Romanian, Slovak, Slovene, Spanish, Swedish, Turkish.

## ⚙️ The Model
1. **Text Preprocessing:** The input text is converted to lowercase, dashes are replaced with whitespaces and all digits and punctuation marks are removed.
2. **Tokenization:** Documents are transformed into sparse count matrices, ignoring words that appear less than 3 times
3. **Prediction:** The custom `myMNB` class calculates the posterior probabilities for each language and predicts the most likely language.

## 🏆 Performance
The model achieved a **97.83% accuracy**  on the test set. 

## 📂 Repository Structure
* `app.py`: The Streamlit web application script.
* `model.py`: Contains the custom `myMNB` class and text preprocessing function.
* `It's_All_Greek_To_Me.ipynb`: The Jupyter Notebook containing exploratory data analysis, the algorithm built from scratch, and model training/evaluation, alongside with useful explanations.
