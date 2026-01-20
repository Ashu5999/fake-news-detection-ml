"""
Hybrid ML + API Verification
=============================
Combines Machine Learning with NewsAPI verification.
"""

import streamlit as st
import pickle
import re
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import parent directory modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from news_verification import verify_with_newsapi, get_hybrid_verdict

# Note: st.set_page_config is in main streamlit_app.py only

# =============================================================================
# Load ML Model
# =============================================================================

@st.cache_resource
def load_ml_model():
    """Load the trained TF-IDF model and vectorizer."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = pickle.load(open(os.path.join(base_dir, "fake_news_model.pkl"), "rb"))
    tfidf = pickle.load(open(os.path.join(base_dir, "tfidf_vectorizer.pkl"), "rb"))
    return model, tfidf

# =============================================================================
# Text Processing
# =============================================================================

def clean_text(text):
    """Clean and preprocess text for ML model."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_ml_prediction(text, model, tfidf):
    """Get prediction from the ML model."""
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)[0]
    
    try:
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities)
    except AttributeError:
        confidence = 0.85
    
    label = 'REAL' if prediction == 1 else 'FAKE'
    return label, confidence

# =============================================================================
# Main UI
# =============================================================================

st.title("🔗 Hybrid ML + API Verification")
st.markdown("**Combines ML prediction with live news source verification**")

with st.expander("ℹ️ How does this work?", expanded=False):
    st.markdown("""
    This system uses a **two-layer verification approach**:
    
    1. **🤖 Machine Learning**: Analyzes writing patterns
    2. **🌐 NewsAPI**: Searches trusted news sources
    
    **Verdict Logic:**
    - ✅ **VERIFIED REAL**: ML=REAL + Found in sources
    - 🔴 **LIKELY FAKE**: ML=FAKE + Not in sources
    - ⚠️ **UNCERTAIN**: Conflicting signals
    """)

st.divider()

# Load models
model, tfidf = load_ml_model()

# Input Section
col1, col2 = st.columns([3, 1])

with col1:
    headline = st.text_input(
        "📰 News Headline (optional):",
        placeholder="Enter the news headline..."
    )
    
    news_text = st.text_area(
        "📝 News Article Text:",
        height=200,
        placeholder="Paste the full news article here..."
    )

with col2:
    st.markdown("**Options:**")
    use_api = st.checkbox("Enable API Verification", value=True)
    show_details = st.checkbox("Show Detailed Analysis", value=True)
    st.caption("💡 API searches Reuters, BBC, CNN, etc.")

# Analyze Button
if st.button("🔎 Analyze News", type="primary", use_container_width=True):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            ml_prediction, ml_confidence = get_ml_prediction(news_text, model, tfidf)
            
            if use_api:
                api_result = verify_with_newsapi(news_text, headline)
            else:
                api_result = {
                    'status': 'UNABLE_TO_VERIFY',
                    'matches': 0, 'sources': [], 'articles': [],
                    'confidence': 0, 'error': 'API disabled.'
                }
            
            verdict = get_hybrid_verdict(ml_prediction, ml_confidence, api_result)
        
        st.divider()
        st.subheader("📊 Analysis Result")
        
        verdict_text = verdict['verdict']
        emoji = verdict['emoji']
        
        if 'VERIFIED REAL' in verdict_text:
            st.success(f"## {emoji} {verdict_text}")
        elif 'LIKELY FAKE' in verdict_text:
            st.error(f"## {emoji} {verdict_text}")
        else:
            st.warning(f"## {emoji} {verdict_text}")
        
        st.markdown(f"**Combined Confidence:** {verdict['combined_confidence']:.1%}")
        st.info(f"💡 {verdict['explanation']}")
        
        if show_details:
            st.divider()
            st.subheader("🔬 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 ML Model")
                if ml_prediction == 'REAL':
                    st.success(f"**Prediction:** {ml_prediction}")
                else:
                    st.error(f"**Prediction:** {ml_prediction}")
                st.metric("Confidence", f"{ml_confidence:.1%}")
            
            with col2:
                st.markdown("### 🌐 API Verification")
                if api_result['status'] == 'VERIFIED':
                    st.success("**Status:** Found in sources")
                    st.metric("Matches", api_result['matches'])
                elif api_result['status'] == 'NOT_FOUND':
                    st.warning("**Status:** Not found")
                else:
                    st.info("**Status:** Unable to verify")
                    if api_result.get('error'):
                        st.caption(f"⚠️ {api_result['error']}")
                
                if verdict.get('sources'):
                    st.markdown("**Sources:**")
                    for source in verdict['sources'][:5]:
                        st.markdown(f"• {source}")

st.divider()
st.caption("⚠️ This is an AI demo. It analyzes writing patterns, not factual accuracy.")
