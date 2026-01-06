"""
Fake News Detection v2 - Context-Aware Streamlit App
=====================================================
Uses DistilBERT for semantic understanding with confidence scoring.

Run: streamlit run app_v2.py
"""

import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = './fake_news_distilbert'
MIN_WORDS = 10
CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# Load Model (cached)
# =============================================================================

@st.cache_resource
def load_model():
    """Load the fine-tuned DistilBERT model and tokenizer."""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return None, None, "Model not found. Please run training first."
    
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

# =============================================================================
# Prediction Function
# =============================================================================

def predict(text: str, tokenizer, model) -> dict:
    """
    Classify news text with confidence scoring.
    
    Returns:
        dict with prediction, confidence, probabilities, and any warnings
    """
    
    word_count = len(text.split())
    result = {
        'word_count': word_count,
        'prediction': None,
        'confidence': 0.0,
        'prob_real': 0.0,
        'prob_fake': 0.0,
        'warning': None
    }
    
    # Length gating
    if word_count < MIN_WORDS:
        result['warning'] = (
            f"⚠️ Text too short ({word_count} words). "
            f"Reliable analysis requires at least {MIN_WORDS} words."
        )
        result['prediction'] = 'UNCERTAIN'
        return result
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    prob_fake = probs[0].item()
    prob_real = probs[1].item()
    confidence = max(prob_fake, prob_real)
    prediction = 'REAL' if prob_real > prob_fake else 'FAKE'
    
    result['prob_fake'] = prob_fake
    result['prob_real'] = prob_real
    result['confidence'] = confidence
    result['prediction'] = prediction
    
    # Confidence warning
    if confidence < CONFIDENCE_THRESHOLD:
        result['warning'] = (
            "⚠️ Low confidence score. "
            "The model is uncertain about this classification."
        )
    
    return result

# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Fake News Detection v2",
        page_icon="📰",
        layout="centered"
    )
    
    # Header
    st.title("📰 Context-Aware Fake News Detection")
    st.markdown("**v2.0** - Powered by DistilBERT")
    
    # Info box
    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown("""
        This system uses **DistilBERT**, a transformer model that understands 
        context and semantic meaning, unlike simple keyword matching.
        
        **Best for:**
        - Full news articles (50+ words)
        - Checking writing style patterns
        
        **Not designed for:**
        - Fact verification (e.g., "Is X true?")
        - Very short statements
        - Non-news content
        """)
    
    # Load model
    tokenizer, model, error = load_model()
    
    if error:
        st.error(f"❌ {error}")
        st.info("""
        **To train the model:**
        ```bash
        pip install transformers torch
        python 03_train_distilbert.py
        ```
        """)
        return
    
    st.success("✅ Model loaded successfully")
    
    # Input area
    st.subheader("Paste News Article")
    
    news_text = st.text_area(
        label="Enter text to analyze:",
        height=200,
        placeholder="Paste a news article here (at least 10 words for reliable analysis)..."
    )
    
    # Sample buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Load Sample (Real)"):
            st.session_state['sample'] = """
            WASHINGTON (Reuters) - The U.S. Senate passed bipartisan legislation 
            on Tuesday aimed at strengthening cybersecurity measures across federal 
            agencies. The bill, which passed with a 95-3 vote, requires agencies 
            to report cyber incidents within 72 hours and establishes new standards 
            for protecting critical infrastructure.
            """
    with col2:
        if st.button("📋 Load Sample (Fake)"):
            st.session_state['sample'] = """
            BREAKING: Scientists EXPOSED for hiding the TRUTH! You won't believe 
            what they don't want you to know! The mainstream media is covering up 
            the biggest scandal in history. Share this before they delete it! 
            This changes EVERYTHING we thought we knew.
            """
    
    if 'sample' in st.session_state:
        news_text = st.session_state['sample']
        del st.session_state['sample']
        st.rerun()
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not news_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # Get prediction
            with st.spinner("Analyzing..."):
                result = predict(news_text, tokenizer, model)
            
            st.divider()
            
            # Show warning if any
            if result['warning']:
                st.warning(result['warning'])
            
            # Results
            if result['prediction'] == 'UNCERTAIN':
                st.info("🔵 **UNCERTAIN** - Cannot reliably classify short text")
            else:
                # Prediction display
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if result['prediction'] == 'REAL':
                        st.success(f"### 🟢 LIKELY REAL")
                    else:
                        st.error(f"### 🔴 LIKELY FAKE")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col3:
                    st.metric("Words", result['word_count'])
                
                # Probability bar
                st.markdown("**Probability Distribution:**")
                
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.progress(result['prob_real'], text=f"Real: {result['prob_real']:.1%}")
                with prob_col2:
                    st.progress(result['prob_fake'], text=f"Fake: {result['prob_fake']:.1%}")
    
    # Footer
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This is a demo system trained on 2016-2017 data. "
        "It detects writing style patterns, not factual accuracy."
    )

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    main()
