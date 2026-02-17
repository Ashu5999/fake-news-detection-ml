# 📰 Fake News Detection System

AI-powered news analysis with **multiple detection methods** - combining Machine Learning with Live News Verification.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://fake-news-detection-ml-m2vz29vtc7zxt4ckzjwja6.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## � Try It Now

**[Open Live Demo →](https://fake-news-detection-ml-m2vz29vtc7zxt4ckzjwja6.streamlit.app/)**

---

## 🔬 Two Detection Methods

### 1. AI Analyzer (DistilBERT)
Deep learning-powered analysis with explainability:
- Fake/Real classification with confidence scores
- Fake/Real classification with confidence scores.
- Word importance visualization
- Emotion analysis
- Writing style analysis
- Clickbait detection
- Readability scoring

### 2. Hybrid ML + API Verification
Combines ML predictions with live news source verification:
- TF-IDF + Logistic Regression ML model
- **Live NewsAPI verification** against trusted sources (Reuters, BBC, CNN)
- Combined confidence scoring
- Graceful fallback when API unavailable

---

## 🛠️ Architecture

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

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Ashu5999/fake-news-detection-ml.git
cd fake-news-detection-ml
pip install -r requirements.txt
```

### 2. Configure NewsAPI (Optional)

Get a free API key from [NewsAPI.org](https://newsapi.org/register):

```bash
cp .env.example .env
# Edit .env and add: NEWSAPI_KEY=your_api_key_here
```

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 and use the sidebar to switch between detection methods.

---

## 📂 Project Structure

```
fake-news-detection-ml/
├── streamlit_app.py          # Main entry point (landing page)
├── pages/
│   ├── 1_AI_Analyzer.py      # DistilBERT-based analyzer
│   └── 2_Hybrid_Verification.py  # ML + NewsAPI hybrid
├── news_verification.py      # NewsAPI integration module
├── app.py                    # Standalone hybrid app
├── train_model.py            # Model training script
├── fake_news_model.pkl       # Trained TF-IDF model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt          # Dependencies
└── .env.example              # Environment variable template
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| F1 Score | ~0.99 |
| Precision | ~0.99 |
| Recall | ~0.99 |

**Dataset:** 44,898 news articles (2016-2017 political news) + 3,400+ COVID-19 related news

---

## 🛠️ Technologies

| Category | Technologies |
|----------|-------------|
| **ML/NLP** | Scikit-learn, PyTorch, Transformers, DistilBERT |
| **Web** | Streamlit |
| **API** | NewsAPI, Requests |
| **Data** | Pandas, NumPy |

---

## ⚠️ Limitations

- Trained primarily on political news from 2016-2017
- Detects writing style patterns, not factual accuracy
- **Not a fact-checking system** - use for guidance only
- NewsAPI free tier: 100 requests/day

---

## 👤 Author

**Ashutosh Tiwari**  .
AIML Internship Project

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

⭐ **Star this repo if you find it helpful!**
