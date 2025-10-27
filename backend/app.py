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

# Determine which LLM mode to use
LLM_MODE = os.getenv('LLM_MODE', 'mock').lower()  # Options: 'mock', 'huggingface', 'openai'

# Configure based on mode
if LLM_MODE == 'huggingface':
    from huggingface_hub import InferenceClient
    HF_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
    # Use a model that works well with the free API
    HF_MODEL = os.getenv('HUGGING_FACE_MODEL', 'HuggingFaceH4/zephyr-7b-beta')

    if HF_TOKEN:
        hf_client = InferenceClient(token=HF_TOKEN)
        print(f"✅ Using Hugging Face model: {HF_MODEL} with authentication")
    else:
        # Use public API (with rate limits)
        hf_client = InferenceClient()
        print(f"⚠️  Warning: No Hugging Face token provided. Using free public API with rate limits.")
        print(f"   Using model: {HF_MODEL}")
        print(f"   Get a FREE token at: https://huggingface.co/settings/tokens")

elif LLM_MODE == 'openai':
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
            'usa': 'The capital of the United States is Washington, D.C. It was established as the capital in 1790.',
            'japan': 'The capital of Japan is Tokyo, the largest metropolitan area in the world.',
        },
        'world war': {
            'initial': 'World War 2 ended in 1945. Germany surrendered in May 1945, and Japan surrendered in September 1945 after the atomic bombs were dropped on Hiroshima and Nagasaki.',
            'correction': 'You\'re absolutely right, thank you for the correction! The European theater of World War 2 ended on May 8, 1945, when Germany officially surrendered. This is celebrated as V-E Day (Victory in Europe Day).',
        },
        'mexican war': {
            'initial': 'The Mexican-American War lasted from 1846 to 1848. It ended with the Treaty of Guadalupe Hidalgo on February 2, 1848, in which Mexico ceded about 55% of its territory to the United States, including present-day California, Nevada, Utah, and parts of several other states.',
            'detail': 'The Mexican-American War (1846-1848) was a conflict between the United States and Mexico following the 1845 U.S. annexation of Texas. The war ended on February 2, 1848, with the signing of the Treaty of Guadalupe Hidalgo. Under this treaty, Mexico ceded approximately 525,000 square miles of territory to the U.S. in exchange for $15 million. This included what would become California, Nevada, Utah, most of Arizona, and parts of New Mexico, Colorado, and Wyoming.',
        },
        'civil war': {
            'initial': 'The American Civil War lasted from 1861 to 1865. It ended on April 9, 1865, when Confederate General Robert E. Lee surrendered to Union General Ulysses S. Grant at Appomattox Court House in Virginia.',
            'detail': 'The American Civil War (1861-1865) was fought between the Union (Northern states) and the Confederacy (Southern states) primarily over slavery and states\' rights. The war effectively ended with Lee\'s surrender at Appomattox on April 9, 1865, though some Confederate forces continued fighting for several more weeks. The war resulted in approximately 620,000-750,000 deaths and the abolition of slavery through the 13th Amendment.',
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
        'gravity': {
            'initial': 'Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, gravity gives weight to objects and causes them to fall to the ground when dropped. It also keeps planets in orbit around the Sun.',
            'detail': 'Gravity is described by Newton\'s law of universal gravitation and Einstein\'s general theory of relativity. The force of gravity depends on the masses of the objects and the distance between them. Earth\'s gravity accelerates objects at approximately 9.8 meters per second squared near the surface.',
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
    elif 'mexican war' in last_message or 'mexican-american war' in last_message:
        response_text = responses['mexican war']['detail'] if is_correction else responses['mexican war']['initial']
    elif 'civil war' in last_message and 'american' in last_message:
        response_text = responses['civil war']['detail'] if is_correction else responses['civil war']['initial']
    elif 'world war' in last_message or 'ww2' in last_message or 'wwii' in last_message:
        response_text = responses['world war']['correction'] if is_correction else responses['world war']['initial']
    elif 'planet' in last_message or 'solar system' in last_message:
        response_text = responses['planets']['detail'] if is_correction else responses['planets']['initial']
    elif 'python' in last_message and 'programming' in last_message:
        response_text = responses['python']['negative'] if ('slow' in last_message or 'performance' in last_message) else responses['python']['initial']
    elif 'photosynthesis' in last_message:
        response_text = responses['photosynthesis']['detail'] if ('detail' in last_message or 'explain more' in last_message) else responses['photosynthesis']['initial']
    elif 'gravity' in last_message:
        response_text = responses['gravity']['detail'] if is_correction else responses['gravity']['initial']

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
    """Generate a more intelligent generic response for unknown questions"""
    # Better topic extraction
    question_lower = question.lower()
    words = question_lower.split()

    # Remove stop words more effectively
    stop_words = {'what', 'when', 'where', 'which', 'who', 'how', 'why', 'does', 'did', 'do',
                  'is', 'are', 'was', 'were', 'the', 'a', 'an', 'about', 'year', 'end'}
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]

    # Try to extract the main topic
    topic = ' '.join(meaningful_words[:3]) if meaningful_words else "this topic"

    # Generate contextual responses based on question type
    if any(word in question_lower for word in ['when', 'year', 'date']):
        # Historical/date questions
        templates = [
            f"While I don't have the exact date for {topic} in my current knowledge base, this appears to be a historical question. Historical events often have significant dates that mark important moments. To get the most accurate information, I'd recommend checking a reliable historical source or encyclopedia.",
            f"This is a good question about {topic}. For specific dates and historical timelines, it's best to consult authoritative historical resources. Many historical events have complex timelines with multiple significant dates that tell the full story.",
        ]
    elif any(word in question_lower for word in ['who', 'person', 'people']):
        # People/biography questions
        templates = [
            f"This question about {topic} involves people or historical figures. Biographical information and the roles people played in historical events are fascinating subjects that often require detailed research from multiple sources to fully understand.",
            f"Questions about {topic} and the people involved often reveal interesting stories and connections. For detailed biographical information, historical records and scholarly sources provide the most comprehensive answers.",
        ]
    elif any(word in question_lower for word in ['how', 'work', 'process']):
        # Process/explanation questions
        templates = [
            f"This is an interesting question about how {topic} works. Understanding processes often involves breaking them down into steps and seeing how different components interact. For technical or scientific topics, consulting specialized resources can provide detailed explanations.",
            f"The mechanics of {topic} involve various factors and processes. Complex systems often have multiple layers of operation, and understanding each layer helps build a complete picture of how everything works together.",
        ]
    else:
        # General questions
        templates = [
            f"That's an interesting question about {topic}. This subject has various aspects worth exploring, including its background, key concepts, and practical implications. For comprehensive information, consulting specialized resources or experts in this field would provide the most accurate details.",
            f"Good question about {topic}. Many topics like this have rich histories and multiple perspectives to consider. Understanding the context, key facts, and different viewpoints can provide a well-rounded answer.",
            f"This question touches on {topic}, which is a subject that can be explored from multiple angles. The best approach is often to look at historical context, current understanding, and how it relates to other concepts in the field.",
        ]

    return random.choice(templates)


def huggingface_llm(messages):
    """Call Hugging Face Inference API"""
    try:
        # Extract the user's question from messages
        user_messages = [m['content'] for m in messages if m['role'] == 'user']
        if not user_messages:
            return "I'm here to help! What would you like to know?"

        # Use the conversational endpoint which works better with free API
        # Build conversation history
        conversation_history = []
        for msg in messages:
            if msg['role'] == 'user':
                conversation_history.append({"role": "user", "content": msg['content']})
            elif msg['role'] == 'assistant':
                conversation_history.append({"role": "assistant", "content": msg['content']})

        # Call Hugging Face API using chat completion
        response = hf_client.chat_completion(
            messages=conversation_history,
            model=HF_MODEL,
            max_tokens=500,
            temperature=0.7,
        )

        # Extract the response text
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return "I'm currently experiencing rate limits. Please try again in a moment, or consider adding a Hugging Face API token for unlimited access."
        elif "not supported" in error_msg.lower() or "conversational" in error_msg.lower():
            # Fallback to a simpler approach
            return huggingface_llm_simple(messages)
        return f"Error calling Hugging Face: {error_msg}"


def huggingface_llm_simple(messages):
    """Simplified Hugging Face call using text generation"""
    try:
        # Get just the last user message for simplicity
        user_messages = [m['content'] for m in messages if m['role'] == 'user']
        if not user_messages:
            return "I'm here to help! What would you like to know?"

        question = user_messages[-1]

        # Use a simple model that works with text generation
        simple_model = "google/flan-t5-large"

        response = hf_client.text_generation(
            f"Answer this question concisely: {question}",
            model=simple_model,
            max_new_tokens=200,
            temperature=0.7,
        )

        return response.strip()
    except Exception as e:
        return f"I apologize, but I'm having trouble connecting to the AI service. Error: {str(e)}\n\nTip: You can switch back to mock mode by changing LLM_MODE=mock in your .env file, or get a free Hugging Face token at https://huggingface.co/settings/tokens"


def call_llm(messages, model="gpt-3.5-turbo"):
    """Call LLM based on configured mode"""
    if LLM_MODE == 'mock':
        return mock_llm(messages)

    elif LLM_MODE == 'huggingface':
        return huggingface_llm(messages)

    elif LLM_MODE == 'openai':
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"

    else:
        return f"Unknown LLM mode: {LLM_MODE}"


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
    messages = {
        'mock': 'Running in MOCK mode - simulated AI responses, no API needed!',
        'huggingface': f'Running with Hugging Face model: {HF_MODEL}' if LLM_MODE == 'huggingface' else '',
        'openai': 'Running with OpenAI API'
    }

    return jsonify({
        'status': 'healthy',
        'sessions': len(sessions),
        'mode': LLM_MODE,
        'message': messages.get(LLM_MODE, f'Running in {LLM_MODE} mode'),
        'model': HF_MODEL if LLM_MODE == 'huggingface' else None
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
