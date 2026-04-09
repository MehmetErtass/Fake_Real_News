# 📰 Fake & Real News Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-PassiveAggressive-orange" />
  <img src="https://img.shields.io/badge/TF--IDF-Vectorization-blueviolet" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" />
</p>

## 📌 Overview

This project builds a **Natural Language Processing (NLP)** pipeline to automatically classify news articles as either **Fake** or **Real**. It utilizes the **Passive Aggressive Classifier** — an online learning algorithm that is particularly effective for large-scale text classification tasks.

---

## 📂 Dataset

The dataset (`fake_or_real_news.csv`) contains labeled news articles with the following structure:

| Column | Description |
|---|---|
| `title` | Headline of the news article |
| `text` | Full body text of the article |
| `label` | `FAKE` or `REAL` |

---

## 🔍 Project Workflow

### 1. Data Loading & Exploration
- Loaded the CSV dataset using **Pandas**
- Explored class distribution — the dataset is balanced between `FAKE` and `REAL` labels

### 2. Text Preprocessing
Applied the following NLP pipeline to clean the raw text:
- Removed non-alphabetical characters using **Regex**
- Converted all text to **lowercase**
- **Tokenized** the text into individual words
- Applied **WordNet Lemmatization** to normalize word roots (e.g., *running → run*)
- Compiled all cleaned texts into a **corpus**

### 3. Feature Extraction
- Split the corpus into **training (80%)** and **test (20%)** sets
- Applied **TF-IDF Vectorization** (`max_df=0.7`, English stop words removed) to convert text into numerical features

### 4. Model Training
- Trained a **Passive Aggressive Classifier** on the TF-IDF features
- This algorithm stays *passive* on correct classifications and *aggressive* on misclassifications, making it well-suited for text streams

### 5. Evaluation
- Evaluated using **Accuracy Score** and **Confusion Matrix**
- Displayed a side-by-side comparison of original vs. predicted labels

### 6. Custom Prediction
- Accepts **user-defined news text** as input
- Applies the same preprocessing pipeline and predicts whether the news is `FAKE` or `REAL`

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~93–95% |
| Classifier | Passive Aggressive |
| Vectorizer | TF-IDF (max_df=0.7) |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Pandas / NumPy | Data handling |
| NLTK | Lemmatization and NLP utilities |
| scikit-learn | TF-IDF, Passive Aggressive Classifier, evaluation metrics |
| Regex (`re`) | Text cleaning |
| Jupyter Notebook | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn nltk jupyter
```

```python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

### Run the Notebook
```bash
git clone https://github.com/MehmetErtass/Fake_Real_News.git
cd Fake_Real_News
jupyter notebook Fake_News.ipynb
```

---

## 📁 Project Structure

```
Fake_Real_News/
│
├── Fake_News.ipynb            # Main NLP classification notebook
├── fake_or_real_news.csv      # Dataset
└── README.md                  # Project documentation
```

---

## 💡 How Custom Prediction Works

```python
# Enter your own news text
user_input = "Scientists discover water on Mars..."

# Preprocessed → TF-IDF vectorized → Predicted
prediction = pac.predict(tfidf_vectorizer.transform([processed_input]))
# Output: 'REAL' or 'FAKE'
```

---

## 👨‍💻 Author

**Mehmet Ertaş**  
[![GitHub](https://img.shields.io/badge/GitHub-MehmetErtass-181717?logo=github)](https://github.com/MehmetErtass)
