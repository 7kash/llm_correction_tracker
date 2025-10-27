from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import openai
from datetime import datetime
import re
from collections import Counter

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Store conversation sessions in memory (in production, use a database)
sessions = {}


def analyze_text(text):
    """Analyze text for various metrics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Word frequency (top 10 meaningful words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'they', 'we', 'my', 'your', 'his', 'her', 'their', 'our'}
    meaningful_words = [w.lower().strip('.,!?;:') for w in words if w.lower() not in stop_words and len(w) > 2]
    word_freq = Counter(meaningful_words).most_common(10)

    # Sentiment indicators (simple heuristic)
    positive_words = {'good', 'great', 'excellent', 'wonderful', 'amazing', 'best', 'correct', 'right', 'yes', 'perfect', 'true', 'accurate', 'indeed', 'absolutely', 'definitely'}
    negative_words = {'bad', 'wrong', 'incorrect', 'no', 'not', 'never', 'false', 'error', 'mistake', 'unfortunately', 'however', 'but'}
    uncertain_words = {'maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain', 'unclear', 'approximately', 'roughly', 'about'}

    positive_count = sum(1 for w in meaningful_words if w in positive_words)
    negative_count = sum(1 for w in meaningful_words if w in negative_words)
    uncertain_count = sum(1 for w in meaningful_words if w in uncertain_words)

    # Calculate sentiment score (-1 to 1)
    total_sentiment_words = positive_count + negative_count + max(1, uncertain_count)
    sentiment_score = (positive_count - negative_count) / total_sentiment_words if total_sentiment_words > 0 else 0

    # Confidence score (inverse of uncertainty)
    confidence_score = max(0, min(100, 70 - (uncertain_count * 10) + (positive_count * 5)))

    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        'word_frequency': word_freq,
        'sentiment_score': round(sentiment_score, 2),
        'confidence_score': round(confidence_score, 2),
        'positive_indicators': positive_count,
        'negative_indicators': negative_count,
        'uncertainty_indicators': uncertain_count
    }


def call_llm(messages, model="gpt-3.5-turbo"):
    """Call OpenAI API"""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start a new conversation session"""
    data = request.json
    question = data.get('question', '')
    session_id = data.get('session_id', str(datetime.now().timestamp()))

    # Get initial response from LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions clearly and concisely."},
        {"role": "user", "content": question}
    ]

    response = call_llm(messages)
    analysis = analyze_text(response)

    # Create session
    sessions[session_id] = {
        'question': question,
        'interactions': [
            {
                'turn': 0,
                'type': 'initial',
                'response': response,
                'analysis': analysis,
                'messages': messages + [{"role": "assistant", "content": response}],
                'timestamp': datetime.now().isoformat()
            }
        ]
    }

    return jsonify({
        'session_id': session_id,
        'response': response,
        'analysis': analysis,
        'turn': 0
    })


@app.route('/api/correct', methods=['POST'])
def correct():
    """Submit a correction and get updated response"""
    data = request.json
    session_id = data.get('session_id')
    correction = data.get('correction', '')

    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = sessions[session_id]
    last_interaction = session['interactions'][-1]

    # Build conversation with correction
    messages = last_interaction['messages'].copy()
    messages.append({"role": "user", "content": f"Actually, that's not quite right. {correction}"})

    # Get corrected response
    response = call_llm(messages)
    analysis = analyze_text(response)

    # Save interaction
    turn = len(session['interactions'])
    messages.append({"role": "assistant", "content": response})

    session['interactions'].append({
        'turn': turn,
        'type': 'correction',
        'correction': correction,
        'response': response,
        'analysis': analysis,
        'messages': messages,
        'timestamp': datetime.now().isoformat()
    })

    return jsonify({
        'session_id': session_id,
        'response': response,
        'analysis': analysis,
        'turn': turn
    })


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get complete session data"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404

    session = sessions[session_id]

    # Compile analytics across all interactions
    analytics = {
        'word_counts': [i['analysis']['word_count'] for i in session['interactions']],
        'sentence_counts': [i['analysis']['sentence_count'] for i in session['interactions']],
        'avg_sentence_lengths': [i['analysis']['avg_sentence_length'] for i in session['interactions']],
        'sentiment_scores': [i['analysis']['sentiment_score'] for i in session['interactions']],
        'confidence_scores': [i['analysis']['confidence_score'] for i in session['interactions']],
        'turns': list(range(len(session['interactions'])))
    }

    return jsonify({
        'session': session,
        'analytics': analytics
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'sessions': len(sessions)})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
