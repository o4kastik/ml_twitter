#  Sentiment Analysis on Twitter  
###  Machine Learning Semester Project

---

##  Project Overview

This project focuses on **sentiment classification of tweets** into four categories:  
**Positive**, **Negative**, **Neutral**, and **Irrelevant**.  

The main objective is to **develop and evaluate machine learning models** that can automatically detect the **emotional tone** of tweets using **Natural Language Processing (NLP)** techniques.  

By analyzing real-world Twitter data, the project demonstrates how NLP can be applied to understand public opinions, trends, and reactions in social media platforms.

---


 # Objective

Classify tweets into four sentiment categories.

Preprocess raw text data using NLP techniques.

Train and compare two ML models — Logistic Regression and Random Forest.

Evaluate models using standard performance metrics (Accuracy, F1-score, Confusion Matrix).

 # Dataset

Source: Twitter Training Dataset (public dataset for sentiment classification)

Records: ~74,000

# 🧾 Columns

| Column     | Description                                                      |
|-------------|------------------------------------------------------------------|
| id          | Tweet identifier                                                 |
| entity      | Topic or keyword                                                 |
| sentiment   | Sentiment label (Positive, Negative, Neutral, Irrelevant)        |
| text        | Tweet text                                                       |


Class distribution: roughly balanced across four categories.

 # 📁 Project Structure

```
Sentiment-Analysis-Twitter/
├── 📄 Sentiment_Analysis_Twitter.ipynb   # Main Jupyter Notebook with all steps
├── 📄 twitter_training.csv               # Original dataset
├── 📄 report.docx                        # Final report (8–12 pages)
└── 📄 README.md                          # Project documentation
```


 # Methods and Tools

Programming Language: Python 3
Libraries Used:
pandas, numpy, matplotlib, seaborn,scikit-learn, re

# Pipeline Overview:

Data Cleaning: Remove links, mentions, punctuation, lowercase text.

Exploratory Data Analysis (EDA): Visualize class distribution, tweet length, frequent words.

Feature Extraction: Convert text to numeric form using TF-IDF vectorization.

Model Training:

Baseline: Naive Bayes

Improved: Logistic Regression

Evaluation: Accuracy, F1-score, and Confusion Matrix.

# 📊 Results

| Model               | Accuracy | F1-Score |
|----------------------|-----------|-----------|
| Naive Bayes          | ≈ 0.70    | ≈ 0.69    |
| Logistic Regression  | ≈ 0.79    | ≈ 0.78    |


# Key Insights:

Both models successfully learned to distinguish between 4 sentiment types.

Logistic Regression slightly outperformed Naive Bayes on accuracy and F1-score.

Most confusion occurred between “Neutral” and “Irrelevant” tweets — they often contain similar tone.

 # Future Work

Implement deep learning models (BERT, RoBERTa) for improved semantic understanding.

Expand dataset with more recent tweets.

Apply hyperparameter tuning for Random Forest.

Use SMOTE or class weighting if class imbalance increases.

# 👥 Team Members

| Name      | Role              | Contribution                         |
|------------|------------------|--------------------------------------|
| Abay       | Data preprocessing | Cleaning, feature engineering        |
| Shyngys    | Modeling          | ML training and evaluation           |
| Kaisar     | Documentation     | Report and presentation preparation  |

 
