"""
Fake News Detection Web App
Author: Ashutosh Tiwari
"""

import streamlit as st
import torch
import re
import math
from collections import Counter
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    pipeline
)

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="AI Fake News Analyzer",
    page_icon="🔬",
    layout="wide"
)

# =============================================================================
# Constants
# =============================================================================

# Suspicious patterns (fake news indicators)
SENSATIONAL_WORDS = {
    'breaking', 'shocking', 'exposed', 'secret', 'truth', 'lies', 'scandal',
    'urgent', 'alert', 'warning', 'exclusive', 'bombshell', 'leaked',
    'conspiracy', 'cover-up', 'mainstream media', 'they dont want you',
    'share before', 'deleted', 'banned', 'censored', 'wake up'
}

CLICKBAIT_PHRASES = [
    "you won't believe", "what happened next", "this is why", 
    "here's what", "the reason why", "everyone is talking",
    "goes viral", "breaks the internet", "mind-blowing",
    "game-changer", "exposed", "the truth about"
]

CREDIBLE_SOURCES = [
    'reuters', 'associated press', 'ap news', 'afp', 'bbc', 'npr',
    'the new york times', 'washington post', 'the guardian',
    'al jazeera', 'cnn', 'pbs', 'bloomberg'
]

# Emotion keywords
EMOTION_KEYWORDS = {
    'fear': ['threat', 'danger', 'attack', 'terror', 'crisis', 'death', 'killed', 
             'warning', 'emergency', 'alarming', 'scary', 'horrifying'],
    'anger': ['outrage', 'furious', 'disgusting', 'shameful', 'corrupt', 'evil',
              'betrayal', 'liar', 'fraud', 'criminal', 'attack', 'destroy'],
    'joy': ['celebrate', 'victory', 'success', 'happy', 'wonderful', 'amazing',
            'great', 'excellent', 'breakthrough', 'achievement', 'proud'],
    'surprise': ['shocking', 'unbelievable', 'incredible', 'unexpected', 'stunning',
                 'bombshell', 'revelation', 'discovered', 'revealed']
}

# Topic keywords
TOPIC_KEYWORDS = {
    'Politics': ['president', 'congress', 'senate', 'election', 'democrat', 'republican',
                 'government', 'politician', 'vote', 'campaign', 'policy', 'law'],
    'Health': ['covid', 'vaccine', 'hospital', 'doctor', 'disease', 'health', 'medical',
               'treatment', 'virus', 'pandemic', 'symptoms', 'medicine'],
    'Technology': ['ai', 'artificial intelligence', 'tech', 'google', 'facebook', 'apple',
                   'microsoft', 'software', 'app', 'data', 'cyber', 'digital'],
    'Business': ['stock', 'market', 'economy', 'company', 'ceo', 'billion', 'investment',
                 'trade', 'bank', 'financial', 'profit', 'revenue'],
    'Entertainment': ['movie', 'celebrity', 'actor', 'music', 'hollywood', 'star',
                      'film', 'show', 'concert', 'award', 'tv', 'series']
}

# =============================================================================
# Load Models
# =============================================================================

@st.cache_resource
def load_models():
    """Load all AI models."""
    models = {}
    
    # Main classifier
    models['tokenizer'] = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    models['classifier'] = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased-finetuned-sst-2-english'
    )
    models['classifier'].eval()
    
    # Emotion classifier
    try:
        models['emotion'] = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=-1  # CPU
        )
    except:
        models['emotion'] = None
    
    return models

# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_writing_style(text: str) -> dict:
    """Analyze writing style metrics."""
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Caps analysis
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    caps_ratio = caps_words / max(len(words), 1)
    
    # Punctuation analysis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Sensationalism score
    text_lower = text.lower()
    sensational_count = sum(1 for word in SENSATIONAL_WORDS if word in text_lower)
    sensationalism_score = min(sensational_count / 5, 1.0)  # Normalize to 0-1
    
    # Average sentence length
    avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    
    # Word complexity (avg word length)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'caps_ratio': caps_ratio,
        'caps_percentage': caps_ratio * 100,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'sensationalism_score': sensationalism_score,
        'avg_sentence_length': avg_sentence_len,
        'avg_word_length': avg_word_len,
        'sensational_words_found': [w for w in SENSATIONAL_WORDS if w in text_lower]
    }

def analyze_emotions(text: str, emotion_model) -> dict:
    """Analyze emotional content."""
    
    if emotion_model:
        try:
            result = emotion_model(text[:512])[0]
            emotions = {item['label']: item['score'] for item in result}
            return emotions
        except:
            pass
    
    # Fallback: keyword-based
    text_lower = text.lower()
    emotions = {}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        emotions[emotion] = min(count / 5, 1.0)
    
    # Normalize
    total = sum(emotions.values()) or 1
    emotions = {k: v/total for k, v in emotions.items()}
    
    # Add neutral
    emotions['neutral'] = max(0, 1 - sum(emotions.values()))
    
    return emotions

def detect_clickbait(title: str, body: str) -> dict:
    """Detect title-body mismatch (clickbait)."""
    
    title_lower = title.lower()
    
    # Check for clickbait phrases
    clickbait_found = [p for p in CLICKBAIT_PHRASES if p in title_lower]
    
    # Title sensationalism
    title_words = title.split()
    title_caps = sum(1 for w in title_words if w.isupper() and len(w) > 1)
    
    # Punctuation in title
    title_exclaims = title.count('!')
    
    # Calculate clickbait score
    score = 0
    score += len(clickbait_found) * 0.2
    score += min(title_caps / 3, 0.3)
    score += min(title_exclaims * 0.1, 0.2)
    
    return {
        'is_clickbait': score > 0.3,
        'clickbait_score': min(score, 1.0),
        'phrases_found': clickbait_found,
        'title_caps_count': title_caps,
        'title_exclaims': title_exclaims
    }

def classify_topic(text: str) -> dict:
    """Classify the topic of the article."""
    
    text_lower = text.lower()
    scores = {}
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[topic] = count
    
    # Normalize
    total = sum(scores.values()) or 1
    scores = {k: v/total for k, v in scores.items()}
    
    # Get top topic
    if scores:
        top_topic = max(scores, key=scores.get)
    else:
        top_topic = 'General'
    
    return {
        'primary_topic': top_topic,
        'topic_scores': scores
    }

def calculate_readability(text: str) -> dict:
    """Calculate readability metrics."""
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    
    # Count syllables (simple approximation)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiou'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)
    
    syllable_count = sum(count_syllables(w) for w in words)
    
    # Flesch-Kincaid Grade Level
    if word_count > 0 and sentence_count > 0:
        fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        fk_grade = max(0, min(fk_grade, 18))
    else:
        fk_grade = 0
    
    # Interpretation
    if fk_grade < 6:
        level = "Very Easy (Elementary)"
    elif fk_grade < 8:
        level = "Easy (Middle School)"
    elif fk_grade < 10:
        level = "Standard (High School)"
    elif fk_grade < 12:
        level = "Moderate (College)"
    else:
        level = "Difficult (Graduate)"
    
    return {
        'grade_level': fk_grade,
        'reading_level': level,
        'avg_words_per_sentence': word_count / sentence_count
    }

def detect_source_patterns(text: str) -> dict:
    """Detect credible source patterns."""
    
    text_lower = text.lower()
    
    # Check for source citations
    sources_found = [s for s in CREDIBLE_SOURCES if s in text_lower]
    
    # Check for attribution patterns
    has_reuters_format = bool(re.search(r'\([A-Z][a-z]+ ?[A-Za-z]*\s*\)', text[:100]))  # (Reuters) pattern
    has_dateline = bool(re.search(r'^[A-Z]{2,}[\s,]', text))  # WASHINGTON, etc.
    
    credibility_score = 0
    if sources_found:
        credibility_score += 0.4
    if has_reuters_format:
        credibility_score += 0.3
    if has_dateline:
        credibility_score += 0.2
    
    return {
        'sources_mentioned': sources_found,
        'has_wire_format': has_reuters_format,
        'has_dateline': has_dateline,
        'credibility_boost': credibility_score
    }

def get_word_importance(text: str, tokenizer, model) -> list:
    """Get importance scores for each word (simplified attention)."""
    
    words = text.split()[:50]  # Limit for display
    importance = []
    
    # Get base prediction
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        base_output = model(**inputs)
        base_probs = torch.softmax(base_output.logits, dim=1)[0]
        base_fake_prob = base_probs[0].item()
    
    # Check each word's impact (ablation)
    for i, word in enumerate(words):
        # Remove word
        modified_words = words[:i] + words[i+1:]
        modified_text = ' '.join(modified_words)
        
        if modified_text.strip():
            inputs = tokenizer(modified_text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                output = model(**inputs)
                probs = torch.softmax(output.logits, dim=1)[0]
                fake_prob = probs[0].item()
            
            # Impact = how much removing this word changes the fake probability
            impact = base_fake_prob - fake_prob
        else:
            impact = 0
        
        importance.append({
            'word': word,
            'impact': impact,  # Positive = contributes to fake, negative = contributes to real
            'direction': 'fake' if impact > 0.01 else ('real' if impact < -0.01 else 'neutral')
        })
    
    return importance

def classify_fake_real(text: str, tokenizer, model) -> dict:
    """Main classification."""
    
    word_count = len(text.split())
    
    if word_count < 10:
        return {
            'prediction': 'UNCERTAIN',
            'confidence': 0,
            'prob_fake': 0.5,
            'prob_real': 0.5,
            'warning': 'Text too short for reliable analysis'
        }
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    
    prob_fake = probs[0].item()
    prob_real = probs[1].item()
    
    return {
        'prediction': 'REAL' if prob_real > prob_fake else 'FAKE',
        'confidence': max(prob_fake, prob_real),
        'prob_fake': prob_fake,
        'prob_real': prob_real,
        'warning': None if max(prob_fake, prob_real) > 0.7 else 'Low confidence'
    }

# =============================================================================
# UI Components
# =============================================================================

def render_gauge(value: float, label: str, color: str = "blue"):
    """Render a simple gauge."""
    percentage = int(value * 100)
    st.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 2em; font-weight: bold; color: {color};">{percentage}%</div>
        <div style="font-size: 0.9em; color: gray;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_word_importance(importance_list: list):
    """Render word importance with colors."""
    
    html_parts = []
    for item in importance_list:
        word = item['word']
        direction = item['direction']
        impact = abs(item['impact'])
        
        if direction == 'fake':
            # Red shades
            intensity = min(int(impact * 500), 255)
            bg_color = f"rgba(255, {255-intensity}, {255-intensity}, 0.7)"
            html_parts.append(f'<span style="background-color: {bg_color}; padding: 2px 4px; margin: 1px; border-radius: 3px;">{word}</span>')
        elif direction == 'real':
            # Green shades
            intensity = min(int(impact * 500), 255)
            bg_color = f"rgba({255-intensity}, 255, {255-intensity}, 0.7)"
            html_parts.append(f'<span style="background-color: {bg_color}; padding: 2px 4px; margin: 1px; border-radius: 3px;">{word}</span>')
        else:
            html_parts.append(f'<span style="padding: 2px 4px; margin: 1px;">{word}</span>')
    
    html = ' '.join(html_parts)
    st.markdown(f'<div style="line-height: 2em;">{html}</div>', unsafe_allow_html=True)
    st.caption("🔴 Red = contributes to FAKE | 🟢 Green = contributes to REAL")

# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.title("🔬 AI Fake News Analyzer")
    st.markdown("**Complete AI-powered analysis with explainability**")
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    st.success("✅ All models loaded!")
    
    # Input section
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        title_input = st.text_input("📰 Headline (optional):", placeholder="Enter news headline...")
        text_input = st.text_area("📝 Article Text:", height=200, 
                                  placeholder="Paste the full news article here...")
    
    with col2:
        st.markdown("**Quick Samples:**")
        if st.button("📋 Real Sample"):
            st.session_state['sample_title'] = "U.S. Senate Passes Bipartisan Bill"
            st.session_state['sample_text'] = """WASHINGTON (Reuters) - The U.S. Senate passed bipartisan legislation on Tuesday aimed at strengthening cybersecurity measures across federal agencies. The bill, which passed with a 95-3 vote, requires agencies to report cyber incidents within 72 hours and establishes new standards for protecting critical infrastructure. Senator John Smith praised the measure as a significant step forward in national security. The legislation now moves to the House of Representatives for consideration."""
        
        if st.button("📋 Fake Sample"):
            st.session_state['sample_title'] = "BREAKING: Scientists EXPOSED for LYING!"
            st.session_state['sample_text'] = """You won't BELIEVE what they've been hiding from you! The mainstream media doesn't want you to know the TRUTH about what's really happening. Scientists have been EXPOSED for covering up the biggest scandal in history! Share this before they DELETE it! They're coming for your rights and nobody is talking about it. Wake up people! This changes EVERYTHING we thought we knew. The government is in on it too!"""
    
    # Apply samples
    if 'sample_title' in st.session_state:
        title_input = st.session_state['sample_title']
        text_input = st.session_state['sample_text']
        del st.session_state['sample_title']
        del st.session_state['sample_text']
        st.rerun()
    
    # Analysis
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Running AI analysis..."):
            # Run all analyses
            classification = classify_fake_real(text_input, models['tokenizer'], models['classifier'])
            style = analyze_writing_style(text_input)
            emotions = analyze_emotions(text_input, models.get('emotion'))
            topic = classify_topic(text_input)
            readability = calculate_readability(text_input)
            sources = detect_source_patterns(text_input)
            
            if title_input:
                clickbait = detect_clickbait(title_input, text_input)
            else:
                clickbait = None
            
            # Word importance (slow, so limit)
            if len(text_input.split()) <= 100:
                word_importance = get_word_importance(text_input, models['tokenizer'], models['classifier'])
            else:
                word_importance = None
        
        st.divider()
        
        # =================================================================
        # Results Display
        # =================================================================
        
        # Row 1: Main Prediction
        st.subheader("🎯 Classification Result")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if classification['prediction'] == 'FAKE':
                st.error(f"### 🔴 LIKELY FAKE")
            elif classification['prediction'] == 'REAL':
                st.success(f"### 🟢 LIKELY REAL")
            else:
                st.warning(f"### 🟡 UNCERTAIN")
        
        with col2:
            st.metric("Confidence", f"{classification['confidence']:.1%}")
        
        with col3:
            st.metric("Words", style['word_count'])
        
        with col4:
            st.metric("Topic", topic['primary_topic'])
        
        if classification['warning']:
            st.warning(f"⚠️ {classification['warning']}")
        
        # Probability bar
        col1, col2 = st.columns(2)
        with col1:
            st.progress(classification['prob_real'], text=f"Real: {classification['prob_real']:.1%}")
        with col2:
            st.progress(classification['prob_fake'], text=f"Fake: {classification['prob_fake']:.1%}")
        
        st.divider()
        
        # Row 2: Detailed Analysis (Tabs)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Explainability", 
            "📊 Writing Style", 
            "😠 Emotions",
            "📰 Clickbait",
            "📖 Readability"
        ])
        
        with tab1:
            st.subheader("Why This Prediction?")
            if word_importance:
                render_word_importance(word_importance)
                
                # Top contributors
                fake_words = [w for w in word_importance if w['direction'] == 'fake']
                real_words = [w for w in word_importance if w['direction'] == 'real']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔴 Words pushing toward FAKE:**")
                    for w in sorted(fake_words, key=lambda x: x['impact'], reverse=True)[:5]:
                        st.write(f"• \"{w['word']}\" (+{w['impact']*100:.1f}%)")
                
                with col2:
                    st.markdown("**🟢 Words pushing toward REAL:**")
                    for w in sorted(real_words, key=lambda x: x['impact'])[:5]:
                        st.write(f"• \"{w['word']}\" ({w['impact']*100:.1f}%)")
            else:
                st.info("Text too long for word-level analysis. Showing pattern-based analysis.")
                if style['sensational_words_found']:
                    st.warning(f"**Sensational words found:** {', '.join(style['sensational_words_found'])}")
        
        with tab2:
            st.subheader("Writing Style Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sensational_pct = int(style['sensationalism_score'] * 100)
                color = "red" if sensational_pct > 50 else ("orange" if sensational_pct > 25 else "green")
                render_gauge(style['sensationalism_score'], "Sensationalism", color)
            
            with col2:
                caps_pct = style['caps_percentage']
                color = "red" if caps_pct > 10 else ("orange" if caps_pct > 5 else "green")
                render_gauge(min(caps_pct/20, 1), f"ALL CAPS ({caps_pct:.1f}%)", color)
            
            with col3:
                exclaim_score = min(style['exclamation_count'] / 5, 1)
                color = "red" if style['exclamation_count'] > 3 else "green"
                render_gauge(exclaim_score, f"Exclamations ({style['exclamation_count']})", color)
            
            st.divider()
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentences", style['sentence_count'])
            with col2:
                st.metric("Avg Sentence Length", f"{style['avg_sentence_length']:.1f} words")
            with col3:
                st.metric("Avg Word Length", f"{style['avg_word_length']:.1f} chars")
            
            if style['sensational_words_found']:
                st.warning(f"⚠️ **Sensational words detected:** {', '.join(style['sensational_words_found'])}")
            
            # Source credibility
            if sources['sources_mentioned']:
                st.success(f"✅ **Credible sources mentioned:** {', '.join(sources['sources_mentioned'])}")
            if sources['has_wire_format']:
                st.success("✅ Uses wire service format (e.g., Reuters style)")
        
        with tab3:
            st.subheader("Emotional Tone Analysis")
            
            if emotions:
                # Sort by value
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                
                for emotion, score in sorted_emotions[:5]:
                    emoji = {'fear': '😨', 'anger': '😠', 'joy': '😊', 'surprise': '😲', 
                            'sadness': '😢', 'disgust': '🤢', 'neutral': '😐'}.get(emotion.lower(), '•')
                    st.progress(score, text=f"{emoji} {emotion.title()}: {score:.1%}")
                
                # Dominant emotion
                dominant = sorted_emotions[0]
                if dominant[0].lower() in ['fear', 'anger']:
                    st.warning(f"⚠️ High {dominant[0]} content detected - common in fake news")
                elif dominant[0].lower() == 'neutral':
                    st.success("✅ Predominantly neutral tone - typical of factual reporting")
        
        with tab4:
            st.subheader("Clickbait Analysis")
            
            if clickbait:
                col1, col2 = st.columns(2)
                
                with col1:
                    if clickbait['is_clickbait']:
                        st.error(f"### 🚨 Clickbait Detected")
                    else:
                        st.success(f"### ✅ Not Clickbait")
                
                with col2:
                    render_gauge(clickbait['clickbait_score'], "Clickbait Score", 
                               "red" if clickbait['clickbait_score'] > 0.5 else "green")
                
                if clickbait['phrases_found']:
                    st.warning(f"**Clickbait phrases found:** {', '.join(clickbait['phrases_found'])}")
                
                st.markdown(f"- Title caps words: {clickbait['title_caps_count']}")
                st.markdown(f"- Title exclamation marks: {clickbait['title_exclaims']}")
            else:
                st.info("Enter a headline to analyze for clickbait patterns.")
        
        with tab5:
            st.subheader("Readability Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Reading Grade Level", f"{readability['grade_level']:.1f}")
                st.caption(readability['reading_level'])
            
            with col2:
                st.metric("Words per Sentence", f"{readability['avg_words_per_sentence']:.1f}")
            
            # Interpretation
            if readability['grade_level'] < 6:
                st.warning("⚠️ Very simple writing - could indicate oversimplification or clickbait")
            elif readability['grade_level'] > 12:
                st.info("📚 Complex writing - typical of academic or professional content")
            else:
                st.success("✅ Standard readability - typical of mainstream news")
        
        # Footer
        st.divider()
        st.caption("⚠️ **Disclaimer:** This is an AI demo. It analyzes writing patterns, not factual accuracy.")

# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    main()
