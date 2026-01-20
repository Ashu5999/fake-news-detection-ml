"""
News Verification Module
=========================
This module provides external news verification using NewsAPI.
It searches trusted news sources to verify if a news story appears
in credible publications.

Author: Ashutosh Tiwari
"""

import os
import re
import requests
from collections import Counter

# =============================================================================
# Configuration
# =============================================================================

# List of trusted news sources for verification
TRUSTED_SOURCES = [
    'reuters', 'associated-press', 'bbc-news', 'the-new-york-times',
    'the-washington-post', 'the-guardian-uk', 'cnn', 'bloomberg',
    'abc-news', 'cbs-news', 'nbc-news', 'npr', 'al-jazeera-english',
    'the-wall-street-journal', 'usa-today', 'time', 'newsweek'
]

# NewsAPI base URL
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# =============================================================================
# Helper Functions
# =============================================================================

def get_api_key():
    """
    Get NewsAPI key from environment variable.
    
    Returns:
        str: API key or None if not found
    """
    # Try loading from .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    return os.environ.get('NEWSAPI_KEY')


def extract_keywords(text, max_keywords=5):
    """
    Extract key terms from text for search query.
    
    Uses a simple frequency-based approach to find important words,
    filtering out common stop words.
    
    Args:
        text (str): Input text to extract keywords from
        max_keywords (int): Maximum number of keywords to return
        
    Returns:
        list: List of keyword strings
    """
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
        'about', 'after', 'before', 'above', 'below', 'up', 'down', 'out', 'off',
        'over', 'under', 'again', 'further', 'once', 'said', 'says', 'new', 'news'
    }
    
    # Clean and tokenize text
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    
    # Filter stop words and count frequencies
    filtered_words = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered_words)
    
    # Get most common words
    keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return keywords


def build_search_query(text, max_words=10):
    """
    Build a search query from the input text.
    
    Extracts keywords and creates a query string suitable for NewsAPI.
    
    Args:
        text (str): Input news text
        max_words (int): Maximum words in query
        
    Returns:
        str: Search query string
    """
    keywords = extract_keywords(text, max_keywords=max_words)
    
    if not keywords:
        # Fallback: use first few significant words
        words = text.split()[:10]
        keywords = [w for w in words if len(w) > 3][:5]
    
    return ' '.join(keywords)


# =============================================================================
# Main Verification Functions
# =============================================================================

def verify_with_newsapi(text, headline=None):
    """
    Verify news content using NewsAPI.
    
    Searches for similar articles in trusted news sources.
    
    Args:
        text (str): The news article text to verify
        headline (str): Optional headline for more focused search
        
    Returns:
        dict: Verification result with the following structure:
            {
                'status': 'VERIFIED' | 'NOT_FOUND' | 'UNABLE_TO_VERIFY',
                'matches': int,           # Number of matching articles
                'sources': list,          # List of source names
                'articles': list,         # List of matching article details
                'confidence': float,      # Verification confidence (0-1)
                'error': str | None       # Error message if any
            }
    """
    api_key = get_api_key()
    
    # Check if API key is available
    if not api_key:
        return {
            'status': 'UNABLE_TO_VERIFY',
            'matches': 0,
            'sources': [],
            'articles': [],
            'confidence': 0,
            'error': 'NewsAPI key not configured. Set NEWSAPI_KEY environment variable.'
        }
    
    # Build search query - prioritize headline if available
    if headline and len(headline.strip()) > 10:
        query = headline.strip()
    else:
        query = build_search_query(text)
    
    if not query:
        return {
            'status': 'UNABLE_TO_VERIFY',
            'matches': 0,
            'sources': [],
            'articles': [],
            'confidence': 0,
            'error': 'Could not extract keywords for search.'
        }
    
    # Make API request
    try:
        params = {
            'q': query,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 10
        }
        
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)
        
        # Handle API errors
        if response.status_code == 401:
            return {
                'status': 'UNABLE_TO_VERIFY',
                'matches': 0,
                'sources': [],
                'articles': [],
                'confidence': 0,
                'error': 'Invalid NewsAPI key.'
            }
        elif response.status_code == 429:
            return {
                'status': 'UNABLE_TO_VERIFY',
                'matches': 0,
                'sources': [],
                'articles': [],
                'confidence': 0,
                'error': 'API rate limit exceeded. Try again later.'
            }
        elif response.status_code != 200:
            return {
                'status': 'UNABLE_TO_VERIFY',
                'matches': 0,
                'sources': [],
                'articles': [],
                'confidence': 0,
                'error': f'API error: {response.status_code}'
            }
        
        data = response.json()
        
        if data.get('status') != 'ok':
            return {
                'status': 'UNABLE_TO_VERIFY',
                'matches': 0,
                'sources': [],
                'articles': [],
                'confidence': 0,
                'error': data.get('message', 'Unknown API error')
            }
        
        articles = data.get('articles', [])
        
        if not articles:
            return {
                'status': 'NOT_FOUND',
                'matches': 0,
                'sources': [],
                'articles': [],
                'confidence': 0.3,  # Low confidence when not found
                'error': None
            }
        
        # Filter for trusted sources and extract info
        matched_articles = []
        sources_found = set()
        
        for article in articles:
            source_name = article.get('source', {}).get('name', '')
            source_id = article.get('source', {}).get('id', '')
            
            # Check if from trusted source (flexible matching)
            is_trusted = any(
                trusted in source_name.lower() or trusted in (source_id or '').lower()
                for trusted in ['reuters', 'associated press', 'ap ', 'bbc', 'cnn',
                               'nbc', 'abc', 'cbs', 'npr', 'guardian', 'times',
                               'post', 'bloomberg', 'al jazeera', 'usa today']
            )
            
            matched_articles.append({
                'title': article.get('title', ''),
                'source': source_name,
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'is_trusted': is_trusted
            })
            
            if is_trusted:
                sources_found.add(source_name)
        
        # Calculate verification confidence
        trusted_count = len(sources_found)
        total_matches = len(articles)
        
        if trusted_count >= 3:
            confidence = 0.9
            status = 'VERIFIED'
        elif trusted_count >= 1:
            confidence = 0.7
            status = 'VERIFIED'
        elif total_matches >= 3:
            confidence = 0.5
            status = 'VERIFIED'
        else:
            confidence = 0.3
            status = 'NOT_FOUND'
        
        return {
            'status': status,
            'matches': total_matches,
            'sources': list(sources_found),
            'articles': matched_articles[:5],  # Return top 5
            'confidence': confidence,
            'error': None
        }
        
    except requests.exceptions.Timeout:
        return {
            'status': 'UNABLE_TO_VERIFY',
            'matches': 0,
            'sources': [],
            'articles': [],
            'confidence': 0,
            'error': 'API request timed out.'
        }
    except requests.exceptions.RequestException as e:
        return {
            'status': 'UNABLE_TO_VERIFY',
            'matches': 0,
            'sources': [],
            'articles': [],
            'confidence': 0,
            'error': f'Network error: {str(e)}'
        }


def get_hybrid_verdict(ml_prediction, ml_confidence, api_result):
    """
    Combine ML prediction with API verification for final verdict.
    
    Decision Matrix:
    - ML=REAL + API=VERIFIED    → VERIFIED REAL NEWS
    - ML=REAL + API=NOT_FOUND   → UNCERTAIN
    - ML=FAKE + API=VERIFIED    → UNCERTAIN (conflicting signals)
    - ML=FAKE + API=NOT_FOUND   → LIKELY FAKE NEWS
    - API=UNABLE_TO_VERIFY      → Use ML result with warning
    
    Args:
        ml_prediction (str): 'REAL' or 'FAKE'
        ml_confidence (float): ML model confidence (0-1)
        api_result (dict): Result from verify_with_newsapi()
        
    Returns:
        dict: Final verdict with explanation
    """
    api_status = api_result['status']
    api_confidence = api_result.get('confidence', 0)
    
    # Handle API failure - fall back to ML only
    if api_status == 'UNABLE_TO_VERIFY':
        if ml_prediction == 'REAL':
            verdict = 'LIKELY REAL NEWS (API Unavailable)'
            emoji = '🟡'
        else:
            verdict = 'LIKELY FAKE NEWS (API Unavailable)'
            emoji = '🟠'
        
        return {
            'verdict': verdict,
            'emoji': emoji,
            'explanation': f"ML prediction: {ml_prediction} ({ml_confidence:.1%} confidence). "
                          f"API verification unavailable: {api_result.get('error', 'Unknown error')}",
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'api_status': api_status,
            'sources': [],
            'combined_confidence': ml_confidence * 0.7  # Reduced confidence without API
        }
    
    # Combine ML and API results
    if ml_prediction == 'REAL' and api_status == 'VERIFIED':
        # Both agree it's real - high confidence
        verdict = 'VERIFIED REAL NEWS'
        emoji = '✅'
        combined_confidence = (ml_confidence + api_confidence) / 2
        explanation = (f"ML model predicts REAL ({ml_confidence:.1%}). "
                      f"Found in {len(api_result['sources'])} trusted source(s).")
    
    elif ml_prediction == 'REAL' and api_status == 'NOT_FOUND':
        # ML says real but can't verify externally
        verdict = 'UNCERTAIN – NEEDS MANUAL VERIFICATION'
        emoji = '⚠️'
        combined_confidence = ml_confidence * 0.5
        explanation = (f"ML model predicts REAL ({ml_confidence:.1%}), but "
                      f"no matching articles found in trusted news sources.")
    
    elif ml_prediction == 'FAKE' and api_status == 'VERIFIED':
        # Conflict: ML says fake but found in sources
        verdict = 'UNCERTAIN – NEEDS MANUAL VERIFICATION'
        emoji = '⚠️'
        combined_confidence = 0.5
        explanation = (f"Conflicting signals: ML predicts FAKE ({ml_confidence:.1%}), but "
                      f"similar content found in news sources. Manual review recommended.")
    
    else:  # ml_prediction == 'FAKE' and api_status == 'NOT_FOUND'
        # Both agree it's likely fake
        verdict = 'LIKELY FAKE NEWS'
        emoji = '🔴'
        combined_confidence = (ml_confidence + (1 - api_confidence)) / 2
        explanation = (f"ML model predicts FAKE ({ml_confidence:.1%}). "
                      f"No matching articles found in trusted news sources.")
    
    return {
        'verdict': verdict,
        'emoji': emoji,
        'explanation': explanation,
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence,
        'api_status': api_status,
        'sources': api_result.get('sources', []),
        'articles': api_result.get('articles', []),
        'combined_confidence': combined_confidence
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Test the module
    test_text = """
    The Federal Reserve raised interest rates by 0.25% today, 
    marking the tenth consecutive rate increase since March 2022.
    Fed Chair Jerome Powell emphasized the commitment to controlling inflation.
    """
    
    print("Testing News Verification Module")
    print("=" * 50)
    print(f"API Key configured: {'Yes' if get_api_key() else 'No'}")
    print(f"\nTest text keywords: {extract_keywords(test_text)}")
    print(f"Search query: {build_search_query(test_text)}")
    
    if get_api_key():
        print("\nVerifying with NewsAPI...")
        result = verify_with_newsapi(test_text)
        print(f"Status: {result['status']}")
        print(f"Matches: {result['matches']}")
        print(f"Sources: {result['sources']}")
        if result['error']:
            print(f"Error: {result['error']}")
    else:
        print("\nSkipping API test - no API key configured.")
