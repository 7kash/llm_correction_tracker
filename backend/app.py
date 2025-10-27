from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import random
from datetime import datetime
import re
from collections import Counter

load_dotenv()

app = Flask(__name__)
CORS(app)

# Check if we should use mock mode (no API key required)
USE_MOCK_MODE = os.getenv('USE_MOCK_MODE', 'true').lower() == 'true'

# Configure OpenAI only if not in mock mode
if not USE_MOCK_MODE:
    import openai
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


def mock_llm(messages):
    """Mock LLM that generates simulated responses without API calls"""
    # Get the last user message
    user_messages = [m for m in messages if m['role'] == 'user']
    if not user_messages:
        return "I'm here to help! What would you like to know?"

    last_message = user_messages[-1]['content'].lower()
    is_correction = 'actually' in last_message or 'wrong' in last_message or 'not quite' in last_message

    # Knowledge base for common questions
    responses = {
        'capital': {
            'france': 'The capital of France is Paris. Paris is known as the "City of Light" and is famous for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.',
            'germany': 'The capital of Germany is Berlin. It became the capital after German reunification in 1990.',
            'spain': 'The capital of Spain is Madrid, located in the center of the country.',
            'italy': 'The capital of Italy is Rome, also known as the "Eternal City".',
        },
        'world war': {
            'initial': 'World War 2 ended in 1945. Germany surrendered in May 1945, and Japan surrendered in September 1945 after the atomic bombs were dropped on Hiroshima and Nagasaki.',
            'correction': 'You\'re absolutely right, thank you for the correction! The European theater of World War 2 ended on May 8, 1945, when Germany officially surrendered. This is celebrated as V-E Day (Victory in Europe Day).',
        },
        'planets': {
            'initial': 'There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006.',
            'detail': 'The eight planets in our solar system can be divided into two groups. The inner rocky planets are Mercury, Venus, Earth, and Mars. The outer gas giants are Jupiter, Saturn, Uranus, and Neptune. Each planet has unique characteristics and orbits the Sun at different distances.',
        },
        'python': {
            'initial': 'Yes, Python is an excellent programming language! It\'s known for being beginner-friendly with a simple, readable syntax. Python is versatile and widely used in web development, data science, machine learning, automation, and scientific computing.',
            'negative': 'That\'s a fair point. Python can indeed be slower than compiled languages like C++ or Rust for computationally intensive tasks. However, for many applications, the development speed and rich ecosystem of libraries make Python a practical choice. You can also use tools like NumPy, Cython, or PyPy to optimize performance-critical sections.',
        },
        'photosynthesis': {
            'initial': 'Photosynthesis is the process by which plants convert light energy into chemical energy. Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.',
            'detail': 'Photosynthesis occurs in the chloroplasts of plant cells. The process has two main stages: the light-dependent reactions (which occur in the thylakoid membranes and produce ATP and NADPH) and the light-independent reactions or Calvin cycle (which occur in the stroma and use ATP and NADPH to convert CO2 into glucose).',
        },
    }

    # Determine response based on question content
    response_text = None

    # Check for specific topics
    if 'capital' in last_message:
        for country, answer in responses['capital'].items():
            if country in last_message:
                response_text = answer
                break
    elif 'world war' in last_message or 'ww2' in last_message or 'wwii' in last_message:
        response_text = responses['world war']['correction'] if is_correction else responses['world war']['initial']
    elif 'planet' in last_message or 'solar system' in last_message:
        response_text = responses['planets']['detail'] if is_correction else responses['planets']['initial']
    elif 'python' in last_message and 'programming' in last_message:
        response_text = responses['python']['negative'] if ('slow' in last_message or 'performance' in last_message) else responses['python']['initial']
    elif 'photosynthesis' in last_message:
        response_text = responses['photosynthesis']['detail'] if ('detail' in last_message or 'explain more' in last_message) else responses['photosynthesis']['initial']

    # Generate generic responses if no specific match
    if not response_text:
        if is_correction:
            corrections = [
                "Thank you for that clarification! You're absolutely right. Let me provide a more accurate response based on your feedback.",
                "I appreciate the correction! You make a good point. Here's an updated answer that takes your input into account.",
                "You're correct, I should be more precise. Based on what you've told me, here's a better explanation.",
            ]
            response_text = random.choice(corrections) + " " + _generate_generic_response(last_message)
        else:
            response_text = _generate_generic_response(last_message)

    # Add some variation in response style
    if len(messages) > 3:  # After several interactions, be more confident
        response_text = response_text.replace("perhaps", "").replace("might", "will").replace("could", "can")

    return response_text


def _generate_generic_response(question):
    """Generate a generic response for unknown questions"""
    templates = [
        "That's an interesting question! Based on common knowledge, {topic} is a complex subject that involves multiple factors. Generally speaking, the key aspects include understanding the fundamentals and how they apply in practice.",
        "Great question! {topic} is something that has been studied extensively. The main points to consider are the historical context, current understanding, and practical applications.",
        "To answer your question about {topic}, we need to consider several important factors. The primary consideration is how different elements interact and influence the overall outcome.",
    ]

    # Try to extract a topic from the question
    words = question.split()
    meaningful_words = [w for w in words if len(w) > 4 and w not in ['what', 'when', 'where', 'which', 'how', 'why', 'does', 'is', 'are', 'the', 'about']]
    topic = meaningful_words[0] if meaningful_words else "this topic"

    return random.choice(templates).format(topic=topic)


def call_llm(messages, model="gpt-3.5-turbo"):
    """Call LLM (either mock or real OpenAI API)"""
    if USE_MOCK_MODE:
        return mock_llm(messages)

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
    return jsonify({
        'status': 'healthy',
        'sessions': len(sessions),
        'mode': 'mock' if USE_MOCK_MODE else 'openai',
        'message': 'Running in MOCK mode - no API key needed!' if USE_MOCK_MODE else 'Running with OpenAI API'
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
