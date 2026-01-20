# 📰 Fake News Detection using NLP

A machine learning project that detects fake news using **Natural Language Processing** and **Deep Learning**, enhanced with **Live News API Verification**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🆕 Hybrid ML + API Verification

This project features a **two-layer verification system** that combines ML predictions with live news source verification:

1. **🤖 Machine Learning Model**: Analyzes writing patterns using TF-IDF + Logistic Regression
2. **🌐 NewsAPI Integration**: Cross-checks predictions against trusted news sources (Reuters, BBC, CNN, etc.)

### Why Hybrid?

ML models can produce false positives. The API layer acts as a safety net by verifying if the story exists in credible publications.

### Architecture

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

### 1. Clone and Install

```bash
git clone https://github.com/Ashu5999/fake-news-detection-ml.git
cd fake-news-detection-ml
pip install -r requirements.txt
```

### 2. Configure NewsAPI (Recommended)

Get a free API key from [NewsAPI.org](https://newsapi.org/register) (100 requests/day).

```bash
# Create .env file from template
cp .env.example .env

# Edit .env and add your key
NEWSAPI_KEY=your_api_key_here
```

### 3. Run the Application

```bash
# Hybrid ML + API app (recommended)
streamlit run app.py

# Advanced DistilBERT app (more features, no API)
streamlit run streamlit_app.py
```

## 📂 Project Structure

```
fake-news-detection-ml/
├── app.py                    # 🔥 Hybrid ML + API verification app
├── news_verification.py      # NewsAPI integration module
├── streamlit_app.py          # Advanced DistilBERT web app
├── train_model.py            # Model training script
├── prepare_data.py           # Data preprocessing script
├── fake_news_model.pkl       # Trained TF-IDF model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt          # Dependencies
├── .env.example              # Environment variable template
├── Fake.csv                  # Fake news dataset (~63MB)
├── True.csv                  # Real news dataset (~54MB)
└── README.md
```

## 🔬 Features

### Hybrid App (`app.py`)
- **ML Prediction** with confidence scores
- **Live API Verification** via NewsAPI
- **Trusted Source Matching** (Reuters, BBC, Al Jazeera, etc.)
- **Graceful Fallback** when API unavailable
- **Related Article Links** for manual verification

### Advanced App (`streamlit_app.py`)
- **DistilBERT** transformer-based classification
- **Word Importance Visualization** - see which words triggered predictions
- **Emotion Analysis** - detects fear, anger, surprise
- **Writing Style Analysis** - measures sensationalism
- **Clickbait Detection** - analyzes headline patterns
- **Readability Scoring** - text complexity metrics

## 📊 Dataset

The model is trained on:
- **44,898** news articles (2016-2017 political news)
- **3,400+** COVID-19 related news (2020)

## 🛠️ Technologies

| Category | Technologies |
|----------|-------------|
| **ML/NLP** | Scikit-learn, PyTorch, Transformers, DistilBERT |
| **Web** | Streamlit |
| **API** | NewsAPI, Requests |
| **Data** | Pandas, NumPy |
| **Config** | python-dotenv |

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

| Tier | Limit |
|------|-------|
| Free | 100 requests/day |
| Developer | 500 requests/day |

> **Note:** The app works without an API key (ML-only mode)

## ⚠️ Limitations

- ML model trained primarily on political news from 2016-2017
- Detects writing style patterns, not factual accuracy
- **Not a fact-checking system** - use for guidance only
- NewsAPI free tier has daily request limits
- API verification depends on news coverage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 👤 Author

**Ashutosh Tiwari**  
AIML Internship Project

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **Star this repo if you find it helpful!**
