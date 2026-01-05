# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview
Fake news spreads rapidly on digital platforms and can mislead public opinion.  
This project aims to automatically classify news articles as Fake or Real using Machine Learning and Natural Language Processing (NLP).

The model is trained on a real-world news dataset and achieves ~99% accuracy using TF-IDF feature extraction and Logistic Regression.

---

## 🎯 Objectives
- Detect fake and real news articles automatically
- Apply NLP techniques to clean and process text data
- Convert text into numerical features using TF-IDF
- Train and evaluate a machine learning classification model
- Save and reload the trained model for future predictions

---

## 🛠️ Technologies & Tools Used
- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Pickle (Model Saving)

---

## 📂 Project Structure
fake-news-detection-ml/
│
├── 01_load_and_inspect_data.ipynb
├── 02_prediction_and_saving_model.ipynb
├── Fake.csv
├── True.csv
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
└── README.md

---

## 📊 Dataset Description
The dataset contains labeled news articles:

- Fake.csv – Fake news articles
- True.csv – Real news articles

Columns:
- title – News headline
- text – Full news content
- subject – Category
- date – Publication date

Total samples: 44,898 news articles

---

## 🔄 Methodology
1. Data Loading & Inspection
   - Loaded Fake and Real news datasets
   - Checked data shape and missing values

2. Data Preprocessing
   - Converted text to lowercase
   - Removed punctuation, numbers, and extra spaces

3. Feature Extraction
   - Used TF-IDF Vectorizer to convert text into numerical features

4. Model Training
   - Trained Logistic Regression classifier
   - Split data into 80% training and 20% testing

5. Model Evaluation
   - Evaluated using Accuracy, Precision, Recall, and F1-score

---

## ✅ Model Performance
- Accuracy: ~99%
- Balanced performance on both Fake and Real news classes

---

## 🔮 Prediction Example
sample_news = "The government announced a new education policy today."
prediction = predict_news(sample_news)

Output:
REAL NEWS

---

## 💾 Model Persistence
The trained model and TF-IDF vectorizer are saved using Pickle.

Saved files:
- fake_news_model.pkl
- tfidf_vectorizer.pkl

This allows predictions without retraining the model.

---

## ⚠️ Limitations
- Dataset mainly contains political news from 2016–2017
- Model may misclassify modern or unrelated news topics
- Performance depends on similarity between training data and input news

---

## 🚀 Future Improvements
- Build a Streamlit-based web interface
- Train with a more diverse and recent dataset
- Experiment with advanced models (Naive Bayes, SVM, LSTM)
- Deploy as a web API

---

## 📌 How to Run the Project
1. Clone the repository:
   git clone https://github.com/your-username/fake-news-detection-ml.git

2. Open Jupyter Notebook:
   jupyter notebook

3. Run notebooks in order:
   - 01_load_and_inspect_data.ipynb
   - 02_prediction_and_saving_model.ipynb

---

## 👤 Author
Ashutosh Tiwari  
AIML Internship Project

⭐ If you like this project, feel free to star the repository!
