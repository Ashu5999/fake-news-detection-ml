# 📰 Fake News Detection using NLP

A machine learning project that detects fake news using **Natural Language Processing** and **Deep Learning**.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run streamlit_app.py
```

## 📂 Project Structure

```
fake-news-detection-ml/
├── streamlit_app.py          # Main web application
├── train_model.py            # Model training script
├── prepare_data.py           # Data preprocessing script
├── app.py                    # Simple TF-IDF based classifier
├── train_on_colab.ipynb      # Google Colab training notebook
├── requirements.txt          # Dependencies
├── Fake.csv                  # Fake news dataset
├── True.csv                  # Real news dataset
└── README.md
```

## 🔬 Features

- **Fake/Real Classification** with confidence scores
- **Explainability** - see which words triggered the prediction
- **Emotion Analysis** - detects fear, anger, surprise in text
- **Writing Style Analysis** - measures sensationalism, caps usage
- **Clickbait Detection** - analyzes headline patterns
- **Readability Scoring** - calculates text complexity

## 📊 Dataset

The model is trained on:
- **44,898** news articles (2016-2017 political news)
- **3,400+** COVID-19 related news (2020)

## 🛠️ Technologies

- Python 3.9+
- PyTorch & Transformers (DistilBERT)
- Streamlit
- Scikit-learn
- Pandas, NumPy

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| F1 Score | ~0.99 |
| Precision | ~0.99 |
| Recall | ~0.99 |

## ⚠️ Limitations

- Trained primarily on political news from 2016-2017
- Detects writing style patterns, not factual accuracy
- Not a fact-checking system

## 👤 Author

**Ashutosh Tiwari**  
AIML Internship Project

---
⭐ Star this repo if you find it helpful!
