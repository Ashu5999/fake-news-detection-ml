"""
Fake News Detection - Multi-Page App
=====================================
Main entry point for Streamlit multi-page application.

Author: Ashutosh Tiwari
"""

import streamlit as st

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Main Landing Page
# =============================================================================

st.title("📰 Fake News Detection System")
st.markdown("### AI-powered news analysis with multiple detection methods")

st.divider()

# Feature Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔬 AI Analyzer
    
    **Deep Learning powered analysis using DistilBERT**
    
    Features:
    - Fake/Real classification with confidence scores
    - Word importance visualization
    - Emotion analysis
    - Writing style analysis
    - Clickbait detection
    - Readability scoring
    
    👉 Select **"🔬 AI Analyzer"** from the sidebar
    """)

with col2:
    st.markdown("""
    ### 🔗 Hybrid Verification
    
    **ML + NewsAPI for double verification**
    
    Features:
    - TF-IDF + Logistic Regression ML model
    - Live NewsAPI verification
    - Searches trusted sources (Reuters, BBC, CNN)
    - Combined confidence scoring
    - Graceful fallback when API unavailable
    
    👉 Select **"🔗 Hybrid Verification"** from the sidebar
    """)

st.divider()

# Quick Start
st.subheader("🚀 Quick Start")
st.markdown("""
1. **Choose a detection method** from the sidebar on the left
2. **Paste a news article** or use the sample buttons
3. **Click Analyze** to get the verdict
""")

# Info box
st.info("""
💡 **Tip:** The Hybrid Verification method combines ML predictions with live news source checks 
for more reliable results. It can catch cases where ML alone might give false positives.
""")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    ⚠️ <b>Disclaimer:</b> This tool analyzes writing patterns. 
    It is not a fact-checking system.<br>
    <b>Author:</b> Ashutosh Tiwari | <b>AIML Internship Project</b>
</div>
""", unsafe_allow_html=True)
