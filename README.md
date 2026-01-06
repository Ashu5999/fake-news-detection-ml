# 📰 AI Fake News Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-detection-ml-8lkpepmx8xejpk235u3yij.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Context-aware fake news detection with AI explainability, emotion analysis, and writing style metrics.**

![Demo Screenshot](https://via.placeholder.com/800x400?text=AI+Fake+News+Analyzer)

---

## 🌟 Features

### 🎯 Core Classification
- **DistilBERT-powered** semantic understanding (not just keyword matching)
- **Confidence scoring** with probability breakdown
- **Length gating** - admits uncertainty on short text instead of guessing

### 🔍 AI Explainability
- **Word-level importance** - see which words pushed toward FAKE/REAL
- **Color-coded visualization** - red for fake indicators, green for real
- **Impact percentages** - quantified contribution of each word

### 📊 Writing Style Analysis
- **Sensationalism score** (0-100%)
- **ALL CAPS detection** with percentage
- **Exclamation abuse** tracking
- **Sensational word detection** (BREAKING, EXPOSED, SECRET, etc.)

### 😠 Emotion Detection
- **Multi-emotion analysis** (fear, anger, joy, surprise, disgust)
- **Emotional tone warnings** - flags high fear/anger content
- **Neutral detection** - identifies factual reporting style

### 📰 Clickbait Detector
- **Headline vs body analysis**
- **Clickbait phrase detection** ("You won't believe", etc.)
- **Title sensationalism metrics**

### 📖 Readability Analysis
- **Flesch-Kincaid Grade Level**
- **Reading complexity warnings**
- **Average sentence/word length metrics**

### 🔗 Source Credibility
- **Wire service format detection** (Reuters, AP style)
- **Credible source mentions** (BBC, NYT, etc.)
- **Dateline pattern recognition**

---

## 🚀 Quick Start

### Option 1: Ultimate AI Analyzer (Recommended)
```bash
# Install dependencies
pip install streamlit transformers torch

# Run the full-featured analyzer
streamlit run app_ultimate.py
```

### Option 2: Simple Demo (No training required)
```bash
streamlit run app_v2_demo.py
```

### Option 3: Train Your Own Model
```bash
# Install dependencies
pip install -r requirements_v2.txt

# Train DistilBERT on your data (~20min GPU, ~2hrs CPU)
python 03_train_distilbert.py

# Run production app
streamlit run app_v2.py
```

### Option 4: Train on Google Colab (Free GPU)
1. Upload `train_on_colab.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Upload your CSV files when prompted
4. Download the trained model

---

## 📂 Project Structure

```
fake-news-detection-ml/
│
├── 🎯 MAIN APPS
│   ├── app_ultimate.py          # Full AI Analyzer (all features) ⭐
│   ├── app_v2.py                # Production DistilBERT app
│   ├── app_v2_demo.py           # Quick demo (pre-trained model)
│   └── app.py                   # Original v1 (TF-IDF)
│
├── 🧠 TRAINING
│   ├── 03_train_distilbert.py   # DistilBERT training script
│   ├── train_on_colab.ipynb     # Google Colab notebook
│   ├── 01_load_and_inspect_data.ipynb  # Data exploration (v1)
│   └── 02_prediction_and_saving_model.ipynb  # Model testing (v1)
│
├── 📊 DATA
│   ├── Fake.csv                 # 23,481 fake news articles
│   └── True.csv                 # 21,417 real news articles
│
├── 🤖 MODELS
│   ├── fake_news_distilbert/    # Trained DistilBERT (after training)
│   ├── fake_news_model.pkl      # Trained Logistic Regression (v1)
│   └── tfidf_vectorizer.pkl     # TF-IDF vectorizer (v1)
│
└── 📋 CONFIG
    ├── requirements.txt         # v1 dependencies
    └── requirements_v2.txt      # v2 dependencies
```

---

## 🔄 Version Comparison

| Feature | v1 (TF-IDF) | v2 (DistilBERT) | Ultimate |
|---------|-------------|-----------------|----------|
| **Classification** | ✅ | ✅ | ✅ |
| **Confidence Score** | ❌ | ✅ | ✅ |
| **Length Gating** | ❌ | ✅ | ✅ |
| **Explainability** | ❌ | ❌ | ✅ |
| **Emotion Analysis** | ❌ | ❌ | ✅ |
| **Writing Style** | ❌ | ❌ | ✅ |
| **Clickbait Detection** | ❌ | ❌ | ✅ |
| **Readability Score** | ❌ | ❌ | ✅ |
| **Source Credibility** | ❌ | ❌ | ✅ |
| **Topic Classification** | ❌ | ❌ | ✅ |

---

## 🧪 How It Works

### 1. Semantic Understanding (DistilBERT)
Unlike TF-IDF which only counts words, DistilBERT understands context:
- "Bank" in "river bank" ≠ "Bank" in "money bank"
- Handles unseen words through subword tokenization
- Pre-trained on modern web text (knows current entities)

### 2. Multi-Layer Defense
```
Input Text
    ↓
[Length Check] → Too short? → Return "UNCERTAIN"
    ↓
[DistilBERT Classification] → Get probabilities
    ↓
[Confidence Check] → Low confidence? → Add warning
    ↓
[Style Analysis] → Check sensationalism
    ↓
[Emotion Analysis] → Flag fear/anger content
    ↓
Final Result + Explanation
```

### 3. Explainability
Uses ablation-based importance:
1. Get base prediction probability
2. Remove each word, re-predict
3. Measure probability change
4. Color-code by impact

---

## 📊 Model Performance

| Metric | v1 (TF-IDF) | v2 (DistilBERT) |
|--------|-------------|-----------------|
| **Accuracy** | ~99% | ~99%+ |
| **F1 Score** | ~0.99 | ~0.99 |
| **Inference Time** | ~1ms | ~50-100ms |
| **Model Size** | 185 KB | 260 MB |

---

## ⚠️ Limitations

> **Be honest about these in interviews - it shows technical maturity.**

1. **Not Fact-Checking**: Cannot verify if "India won 2024 Cricket World Cup" is true
2. **Dataset Bias**: Trained on 2016-2017 political news
3. **Style-Based**: Detects writing patterns, not content accuracy
4. **Domain Specific**: May not work well on health/science/sports
5. **Adversarial Weakness**: Sophisticated fake news mimicking real style may fool it

---

## 🛠️ Technologies

- **Python 3.9+**
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - DistilBERT model
- **Streamlit** - Web interface
- **Scikit-learn** - ML utilities
- **Pandas/NumPy** - Data processing

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ **Transfer Learning** with pre-trained transformers
- ✅ **Explainable AI (XAI)** techniques
- ✅ **Multi-task NLP** (classification, emotion, topics)
- ✅ **Production ML** with Streamlit deployment
- ✅ **Text Analytics** (readability, style metrics)

---

## 📈 Future Improvements

- [ ] Add more diverse training data (post-2020 news)
- [ ] Implement cross-domain detection (health, science, sports)
- [ ] Add multi-language support
- [ ] Deploy as REST API
- [ ] Add real-time news source verification

---

## 👤 Author

**Ashutosh Tiwari**  
AIML Internship Project

---

## 📜 License

MIT License - feel free to use and modify!

---

⭐ **If you find this project useful, please star the repository!**
