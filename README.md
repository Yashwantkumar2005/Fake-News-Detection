# Fake News Detection System

A comparative study of machine learning classifiers for fake news detection, with text preprocessing using **Stemming** and **Lemmatization**, and model optimization using **Hyperparameter Tuning (GridSearchCV)**.

---

## Project Overview

This project builds and compares multiple ML classifiers to detect whether a news article is **real or fake**. It evaluates the effect of different NLP preprocessing techniques and hyperparameter tuning strategies on model performance.

---

## Models Compared

| Model | Preprocessing | Hyperparameter Tuning |
|---|---|---|
| Logistic Regression | Lemmatization | GridSearchCV (C, max_iter) |
| Naive Bayes (MultinomialNB) | Lemmatization | GridSearchCV (alpha) |
| Support Vector Machine (SVM) | Lemmatization | GridSearchCV (C, kernel, gamma) |
| Passive Aggressive Classifier | Stemming + Lemmatization | GridSearchCV (C, max_iter) |
| K-Nearest Neighbors (KNN) | Lemmatization | GridSearchCV |

---

## NLP Preprocessing Pipeline

- Removed URLs, mentions, and special characters
- Lowercased text
- **Stemming** using Porter Stemmer
- **Lemmatization** using WordNet Lemmatizer
- Stopword removal (NLTK)
- Feature extraction via **TF-IDF Vectorizer** (max 5000 features)

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (heatmap)
- ROC Curve and AUC Score (all models plotted together)

---

## Project Structure

```
fake-news-detection/
│
├── FakeNewsDetectionComparison_StemmLemmHPT.ipynb   ← Main notebook
├── README.md
├── requirements.txt
└── dataset/
    └── train.csv    ← Dataset (not included — see below)
```

---

## Dataset

This project uses the **Fake News dataset** from Kaggle:  
🔗 [https://www.kaggle.com/c/fake-news/data](https://www.kaggle.com/c/fake-news/data)

Download `train.csv` and place it in the `/content/sample_data/` directory (if using Google Colab) or update the path in the notebook.

**Columns used:**
- `author` — News article author
- `title` — News article title
- `label` — 0 = Real, 1 = Fake

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/)
2. Upload `train.csv` to `/content/sample_data/`
3. Run all cells (`Runtime → Run all`)

### Option 2: Local (Jupyter Notebook)
```bash
# 1. Clone the repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook
```

---

## Requirements

See `requirements.txt` for full list. Key libraries:

- `scikit-learn` — ML models, GridSearchCV, metrics
- `nltk` — Stopwords, Stemmer, Lemmatizer
- `pandas`, `numpy` — Data handling
- `matplotlib`, `seaborn` — Visualization

---

## Results Summary

All models were evaluated on an 80/20 train-test split. ROC curves for all classifiers are plotted together to compare AUC scores visually.