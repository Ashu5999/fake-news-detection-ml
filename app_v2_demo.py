"""
Fake News Detection v2 - DEMO VERSION
======================================
Uses pre-trained DistilBERT (no training required).
This is for testing the UI and confidence features.

Run: streamlit run app_v2_demo.py
"""

import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# =============================================================================
# Configuration
# =============================================================================

MIN_WORDS = 10
CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# Load Pre-trained Model (cached)
# =============================================================================

@st.cache_resource
def load_model():
    """Load pre-trained DistilBERT (not fine-tuned, for demo only)."""
    st.toast("Loading DistilBERT model... (first time takes ~1 min)")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # Use sentiment model as proxy for demo (similar binary classification)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased-finetuned-sst-2-english'
    )
    model.eval()
    return tokenizer, model

# =============================================================================
# Prediction Function
# =============================================================================

def predict(text: str, tokenizer, model) -> dict:
    """Classify news text with confidence scoring."""
    
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
    
    # SST-2 model: 0=negative, 1=positive
    # We map: negative sentiment → likely fake, positive → likely real
    prob_fake = probs[0].item()
    prob_real = probs[1].item()
    confidence = max(prob_fake, prob_real)
    prediction = 'REAL' if prob_real > prob_fake else 'FAKE'
    
    result['prob_fake'] = prob_fake
    result['prob_real'] = prob_real
    result['confidence'] = confidence
    result['prediction'] = prediction
    
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
    st.set_page_config(
        page_title="Fake News Detection v2 DEMO",
        page_icon="📰",
        layout="centered"
    )
    
    st.title("📰 Fake News Detection v2")
    st.caption("🔬 DEMO MODE - Using pre-trained model (not fine-tuned on your data)")
    
    # Warning banner
    st.warning(
        "**Demo Mode**: This uses a pre-trained sentiment model as a proxy. "
        "Results are illustrative only. Train the full model for accurate predictions."
    )
    
    # Load model
    tokenizer, model = load_model()
    st.success("✅ Model loaded!")
    
    # Input
    st.subheader("Paste News Article")
    
    news_text = st.text_area(
        label="Enter text to analyze:",
        height=200,
        placeholder="Paste a news article here..."
    )
    
    # Sample buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Sample (Real-style)"):
            st.session_state['sample'] = """
            WASHINGTON (Reuters) - The U.S. Senate passed bipartisan legislation 
            on Tuesday aimed at strengthening cybersecurity measures across federal 
            agencies. The bill, which passed with a 95-3 vote, requires agencies 
            to report cyber incidents within 72 hours and establishes new standards 
            for protecting critical infrastructure. Senators praised the measure 
            as a significant step forward in national security.
            """
    with col2:
        if st.button("📋 Sample (Fake-style)"):
            st.session_state['sample'] = """
            BREAKING: Scientists EXPOSED for hiding the TRUTH! You won't BELIEVE 
            what they don't want you to know! The mainstream media is covering up 
            the biggest scandal in history. Share this before they DELETE it! 
            This changes EVERYTHING we thought we knew. They're coming for your 
            rights and nobody is talking about it!
            """
    
    if 'sample' in st.session_state:
        news_text = st.session_state['sample']
        del st.session_state['sample']
        st.rerun()
    
    # Analyze
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not news_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                result = predict(news_text, tokenizer, model)
            
            st.divider()
            
            if result['warning']:
                st.warning(result['warning'])
            
            if result['prediction'] == 'UNCERTAIN':
                st.info("🔵 **UNCERTAIN** - Text too short")
            else:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if result['prediction'] == 'REAL':
                        st.success("### 🟢 LIKELY REAL")
                    else:
                        st.error("### 🔴 LIKELY FAKE")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col3:
                    st.metric("Words", result['word_count'])
                
                # Probabilities
                st.markdown("**Probability:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.progress(result['prob_real'], text=f"Real: {result['prob_real']:.1%}")
                with col2:
                    st.progress(result['prob_fake'], text=f"Fake: {result['prob_fake']:.1%}")
    
    st.divider()
    st.caption("⚠️ Demo only. Train with `python 03_train_distilbert.py` for real predictions.")

if __name__ == '__main__':
    main()
