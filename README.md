# 📰 Fake News Detection using NLP

A machine learning project that detects fake news using **Natural Language Processing** and **Deep Learning**, enhanced with **Live News API Verification**.

## 🆕 Hybrid ML + API Verification

This project now features a **two-layer verification system**:

1. **🤖 Machine Learning Model**: Analyzes writing patterns to classify news as real or fake
2. **🌐 NewsAPI Integration**: Cross-checks predictions against trusted news sources

### How It Works

```
┌─────────────────┐     ┌──────────────────┐
│   User Input    │────▶│   ML Prediction   │
│  (News Article) │     │  (Fake/Real +     │
└─────────────────┘     │   Confidence)     │
         │              └────────┬──────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  NewsAPI Search │────▶│  Hybrid Verdict  │
│ (Trusted Sources)│     │                  │
└─────────────────┘     └──────────────────┘
```

### Verdict Logic

| ML Prediction | API Verification | Final Verdict |
|---------------|------------------|---------------|
| REAL | Found in sources | ✅ VERIFIED REAL NEWS |
| REAL | Not found | ⚠️ UNCERTAIN – NEEDS MANUAL VERIFICATION |
| FAKE | Found in sources | ⚠️ UNCERTAIN – NEEDS MANUAL VERIFICATION |
| FAKE | Not found | 🔴 LIKELY FAKE NEWS |

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure NewsAPI (Optional but Recommended)

Get a free API key from [NewsAPI.org](https://newsapi.org/register) (100 requests/day).

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your key
NEWSAPI_KEY=your_api_key_here
```

### 3. Run the application

```bash
# Hybrid ML + API app (recommended)
streamlit run app.py

# Advanced DistilBERT app
streamlit run streamlit_app.py
```

## 📂 Project Structure

```
fake-news-detection-ml/
├── app.py                    # Hybrid ML + API verification app
├── news_verification.py      # NewsAPI integration module
├── streamlit_app.py          # Advanced DistilBERT web app
├── train_model.py            # Model training script
├── prepare_data.py           # Data preprocessing script
├── fake_news_model.pkl       # Trained TF-IDF model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt          # Dependencies
├── .env.example              # Environment variable template
├── Fake.csv                  # Fake news dataset
├── True.csv                  # Real news dataset
└── README.md
```

## 🔬 Features

### ML-Based Detection
- **Fake/Real Classification** with confidence scores
- **TF-IDF + Logistic Regression** for reliable predictions
- **DistilBERT** advanced model (in streamlit_app.py)

### API Verification (NEW)
- **NewsAPI Integration** for live verification
- **Trusted Source Matching** (Reuters, BBC, CNN, etc.)
- **Graceful Fallback** when API is unavailable
- **Related Article Links** for manual verification

### Analysis Features
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
- Streamlit
- Scikit-learn (TF-IDF + Logistic Regression)
- PyTorch & Transformers (DistilBERT)
- NewsAPI (external verification)
- Pandas, NumPy

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| F1 Score | ~0.99 |
| Precision | ~0.99 |
| Recall | ~0.99 |

## 🔐 API Configuration

### Getting a NewsAPI Key

1. Visit [https://newsapi.org/register](https://newsapi.org/register)
2. Create a free account
3. Copy your API key
4. Create `.env` file: `cp .env.example .env`
5. Add your key: `NEWSAPI_KEY=your_key_here`

### Rate Limits

- **Free tier**: 100 requests/day
- **Developer tier**: 500 requests/day
- The app works without API (ML-only mode)

## ⚠️ Limitations

- ML model trained primarily on political news from 2016-2017
- Detects writing style patterns, not factual accuracy
- **Not a fact-checking system** - use for guidance only
- NewsAPI free tier has daily request limits
- API verification depends on news coverage

## 👤 Author

**Ashutosh Tiwari**  
AIML Internship Project

---
⭐ Star this repo if you find it helpful!
