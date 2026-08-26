# AllGreekToMe

# It's All Greek To Me! 🌍 
**A Custom Language Identification App**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-goes-here.streamlit.app)

## 📌 Overview
This project tackles the task of **Language Identification** for written text[cite: 1]. Instead of relying on off-the-shelf machine learning classifiers, I implemented a **Multinomial Naive Bayes (MNB) algorithm entirely from scratch**[cite: 1]. 

To make the model accessible to the public, I packaged the custom algorithm and deployed it as an interactive web application using **Streamlit**.

## 📊 The Dataset
The model was trained using a balanced subset of the **WiLI-2018 dataset**, which consists of short text extracts from Wikipedia[cite: 1]. 
* **Total Samples:** 10,000 documents (7,000 for training, 3,000 for testing)[cite: 1].
* **Supported Languages (10):** Danish, Dutch, English, Finnish, French, German, Italian, Portuguese, Spanish, and Swedish[cite: 1].

## ⚙️ How It Works
1. **Preprocessing:** The input text is converted to lowercase, dashes are replaced with whitespaces, and all digits and punctuation marks are removed[cite: 1].
2. **Tokenization:** Documents are transformed into sparse count matrices[cite: 1]. By ignoring words that appear less than 3 times (`min_df=3`), the vocabulary is optimized to 18,340 unique words[cite: 1].
3. **Prediction:** The custom `myMNB` class calculates the posterior probabilities for each language and predicts the most likely match[cite: 1].

## 🏆 Performance
The model achieved a remarkable **98.1% accuracy** on the test set[cite: 1]. 
* It excels at separating highly similar language groups (e.g., Danish, Finnish, and Swedish, or Portuguese and Spanish)[cite: 1].
* *Note:* The confusion matrix revealed a slight overlap where some Portuguese documents were incorrectly predicted as English, highlighting an area for future exploration[cite: 1].

## 📂 Repository Structure
* `app.py`: The Streamlit web application script.
* `model.py`: Contains the custom `myMNB` class and text preprocessing logic.
* `It's_All_Greek_To_Me.ipynb`: The original Jupyter Notebook containing exploratory data analysis, the algorithm built from scratch, and model training/evaluation[cite: 1].
* `models/`: Directory containing the serialized `.pkl` files for the trained Naive Bayes model and CountVectorizer.
* `requirements.txt`: Python dependencies needed to run the app.

## 🚀 How to Run Locally
To run this project on your own machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Language-Identifier.git](https://github.com/YourUsername/Language-Identifier.git)
   cd Language-Identifier
