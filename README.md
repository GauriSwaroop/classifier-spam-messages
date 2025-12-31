# classifier-spam-messages# Spam Classification using NLP

## Overview
This project implements a **spam detection system** that classifies SMS messages as **spam** or **ham** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The system applies **TF-IDF vectorization with N-grams** and evaluates multiple models to identify the best-performing classifier.

---

## Dataset
The project uses the **SMS Spam Collection dataset** from the UCI Machine Learning Repository.

- Total messages: 5,574  
- Ham messages: 4,827 (~86.5%)  
- Spam messages: 747 (~13.5%)  
- Each message is labeled as either `spam` or `ham`

The dataset consists of short SMS messages and is widely used as a benchmark for text classification tasks.

---

## Methodology

### 1. Text Preprocessing
- Converted text to lowercase  
- Removed punctuation and special characters  
- Tokenized text into words  
- Removed stopwords  

### 2. Feature Extraction
- Used **TF-IDF Vectorizer**
- Included **unigrams and bigrams (n-gram range = 1–2)** to capture meaningful phrases

### 3. Models Trained
- Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)

---

## Results

| Model | Accuracy | FP | FN |
|------|----------|----|----|
| Naive Bayes | 96.41% | 0 | 40 |
| Logistic Regression | 96.68% | 0 | 37 |
| SVM | 99.10% | 2 | 8 |

SVM achieved the highest accuracy with minimal misclassification.

---

## Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- ROC-AUC Curve  

---

## Technologies Used
- Python  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn  

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-classifier-nlp.git

 NLP-based spam detection using TF-IDF, N-grams, and machine learning models
