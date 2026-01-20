"""
Fake News Detection System - Hybrid ML + API Verification
==========================================================
This app combines Machine Learning prediction with external news 
verification to provide more reliable fake news detection.

Author: Ashutosh Tiwari
"""

import streamlit as st
import pickle
import re

# Import the news verification module
from news_verification import verify_with_newsapi, get_hybrid_verdict

# =============================================================================
# Load ML Model (existing model - unchanged)
# =============================================================================

@st.cache_resource
def load_ml_model():
    """Load the trained TF-IDF model and vectorizer."""
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, tfidf

# =============================================================================
# Text Processing (existing function - unchanged)
# =============================================================================

def clean_text(text):
    """Clean and preprocess text for ML model."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =============================================================================
# ML Prediction
# =============================================================================

def get_ml_prediction(text, model, tfidf):
    """
    Get prediction from the ML model.
    
    Args:
        text: Input news text
        model: Trained classifier
        tfidf: TF-IDF vectorizer
        
    Returns:
        tuple: (prediction_label, confidence_score)
    """
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text])
    
    # Get prediction
    prediction = model.predict(vector)[0]
    
    # Get confidence (probability)
    try:
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities)
    except AttributeError:
        # Model doesn't support predict_proba
        confidence = 0.85  # Default confidence
    
    label = 'REAL' if prediction == 1 else 'FAKE'
    return label, confidence

# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(
    page_title="Fake News Detection - Hybrid System",
    page_icon="🔍",
    layout="wide"
)

# Header
st.title("🔍 Fake News Detection System")
st.markdown("**Hybrid ML + API Verification for Enhanced Accuracy**")

# Info box about the hybrid system
with st.expander("ℹ️ How does this work?", expanded=False):
    st.markdown("""
    This system uses a **two-layer verification approach**:
    
    1. **🤖 Machine Learning Model**: Analyzes writing patterns and style to predict if news is fake or real
    2. **🌐 NewsAPI Verification**: Searches trusted news sources to see if the story appears in credible publications
    
    **Final Verdict Logic:**
    - ✅ **VERIFIED REAL NEWS**: ML predicts REAL + Found in trusted sources
    - 🔴 **LIKELY FAKE NEWS**: ML predicts FAKE + Not found in sources
    - ⚠️ **UNCERTAIN**: Conflicting signals or unable to verify
    """)

st.divider()

# Load models
model, tfidf = load_ml_model()

# Input Section
col1, col2 = st.columns([3, 1])

with col1:
    headline = st.text_input(
        "📰 News Headline (optional):",
        placeholder="Enter the news headline for better verification..."
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
    
    st.markdown("---")
    st.caption("💡 API verification searches trusted sources like Reuters, BBC, CNN, etc.")

# Analyze Button
if st.button("🔎 Analyze News", type="primary", use_container_width=True):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Step 1: ML Prediction
            ml_prediction, ml_confidence = get_ml_prediction(news_text, model, tfidf)
            
            # Step 2: API Verification (if enabled)
            if use_api:
                api_result = verify_with_newsapi(news_text, headline)
            else:
                api_result = {
                    'status': 'UNABLE_TO_VERIFY',
                    'matches': 0,
                    'sources': [],
                    'articles': [],
                    'confidence': 0,
                    'error': 'API verification disabled by user.'
                }
            
            # Step 3: Get Hybrid Verdict
            verdict = get_hybrid_verdict(ml_prediction, ml_confidence, api_result)
        
        st.divider()
        
        # =================================================================
        # Display Results
        # =================================================================
        
        # Main Verdict Card
        st.subheader("📊 Analysis Result")
        
        # Color-coded verdict
        verdict_text = verdict['verdict']
        emoji = verdict['emoji']
        
        if 'VERIFIED REAL' in verdict_text:
            st.success(f"## {emoji} {verdict_text}")
        elif 'LIKELY FAKE' in verdict_text:
            st.error(f"## {emoji} {verdict_text}")
        else:
            st.warning(f"## {emoji} {verdict_text}")
        
        # Confidence and explanation
        st.markdown(f"**Combined Confidence:** {verdict['combined_confidence']:.1%}")
        st.info(f"💡 {verdict['explanation']}")
        
        # Detailed breakdown
        if show_details:
            st.divider()
            st.subheader("🔬 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 ML Model Prediction")
                if ml_prediction == 'REAL':
                    st.success(f"**Prediction:** {ml_prediction}")
                else:
                    st.error(f"**Prediction:** {ml_prediction}")
                st.metric("Confidence", f"{ml_confidence:.1%}")
                st.caption("Based on writing style and pattern analysis")
            
            with col2:
                st.markdown("### 🌐 API Verification")
                api_status = api_result['status']
                
                if api_status == 'VERIFIED':
                    st.success(f"**Status:** Found in news sources")
                    st.metric("Matches", api_result['matches'])
                elif api_status == 'NOT_FOUND':
                    st.warning(f"**Status:** Not found in sources")
                    st.metric("Matches", 0)
                else:
                    st.info(f"**Status:** Unable to verify")
                    if api_result.get('error'):
                        st.caption(f"⚠️ {api_result['error']}")
                
                if verdict.get('sources'):
                    st.markdown("**Trusted sources:**")
                    for source in verdict['sources'][:5]:
                        st.markdown(f"• {source}")
            
            # Show matching articles if available
            if verdict.get('articles'):
                st.divider()
                st.markdown("### 📰 Related Articles Found")
                for i, article in enumerate(verdict['articles'][:3]):
                    with st.container():
                        st.markdown(f"**{i+1}. {article['title']}**")
                        st.caption(f"Source: {article['source']}")
                        if article.get('url'):
                            st.markdown(f"[Read article]({article['url']})")
                        st.markdown("---")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    ⚠️ <b>Disclaimer:</b> This tool analyzes writing patterns and checks news sources. 
    It is not a fact-checking system and should not replace critical thinking.<br>
    <b>Author:</b> Ashutosh Tiwari | <b>AIML Internship Project</b>
</div>
""", unsafe_allow_html=True)
