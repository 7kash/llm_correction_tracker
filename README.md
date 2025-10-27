# 🧠 LLM Learning Visualizer

An **interactive educational tool** that helps non-technical users understand how Large Language Models (LLMs) work by visualizing semantic changes when AI responses are corrected.

![LLM Learning Visualizer](https://img.shields.io/badge/Status-Enhanced-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![React](https://img.shields.io/badge/React-18-blue) ![AI](https://img.shields.io/badge/AI-Semantic_Similarity-purple)

## ✨ What's New - Major Update!

This version includes **comprehensive semantic similarity visualizations** and **interactive educational demos**:

🔥 **Semantic Similarity Heatmaps** - See meaning changes, not just word changes
📊 **Vector Movement Visualization** - Watch AI's "thought process" in 2D semantic space
📝 **Sentence-Level Analysis** - Understand which specific ideas changed
🎓 **Interactive Explainers** - Learn attention & temperature through hands-on demos
📚 **Educational Content** - Theory, formulas, and plain-English explanations for every concept

## 🎯 What Does This Do?

This app lets you:
- **Ask questions** to an AI (OpenAI, Groq, or mock mode)
- **Correct the AI** when it's wrong
- **See beautiful visualizations** showing:
  - **Semantic similarity** between responses (cosine, Jaccard, vector angles)
  - **Response evolution** in 2D semantic space with arrows
  - **Sentence-level changes** - which ideas kept, modified, or replaced
  - Response length, sentiment, and confidence over time

Perfect for teachers, students, or anyone curious about how AI understands and adapts!

## 🚀 Key Features

### 1. **Semantic Similarity Analysis** 🔥 NEW
- **384-dimensional embeddings** using sentence-transformers
- **Cosine similarity matrices** showing how responses relate
- **Plain-English change descriptions**:
  - 🟢 Tiny tweak (>0.95 similarity)
  - 🟡 Small adjustment (>0.85)
  - 🟠 Moderate change (>0.70)
  - 🔴 Big change (>0.40)
  - ⚫ Major rethink (<0.40)

### 2. **Vector Movement in Semantic Space** 🎯 NEW
- **2D projections** using UMAP or PCA
- **Animated arrows** showing the path of semantic change
- **Color gradients** indicating progression
- See how the AI "moves" its understanding through corrections

### 3. **Sentence-Level Comparison** 📝 NEW
- **Heatmap** comparing every sentence to every other
- Reveals which ideas were:
  - ✅ **Kept** (high similarity on diagonal)
  - 🔄 **Modified** (medium similarity off-diagonal)
  - ❌ **Replaced** (low similarity)

### 4. **Multiple Similarity Metrics** 📐 NEW
- **Cosine Similarity**: Semantic meaning (0.0 to 1.0)
- **Jaccard Similarity**: Word overlap percentage
- **Vector Angle**: Geometric difference in degrees
- **Length Change**: Response size change percentage

### 5. **Interactive Educational Demos** 🎓 NEW

Access via `explainers.html` or click the link in the main app!

#### 🗳️ Voting Council
- **Learn**: How attention mechanism works
- **Metaphor**: Context words "vote" for next token
- **Interactive**: Adjust recency and relevance weights
- **Watch**: Real-time probability changes

#### 🌡️ Probability Funnel
- **Learn**: Temperature and softmax function
- **Metaphor**: Temperature controls confidence vs. creativity
- **Interactive**: Slider from 0.1 (confident) to 2.0 (random)
- **Metrics**: Entropy, diversity, top choice probability

### 6. **Comprehensive Educational Content** 📚 NEW
Every visualization includes:
- **📚 Theory Boxes**: Conceptual explanations
- **💡 Explanation Boxes**: How to interpret visualizations
- **🔢 Calculation Boxes**: Formulas with worked examples

### 7. **Beautiful, Functional Design** 🎨
- **Collapsible sections** - No information overload
- **Color-coded themes** - Easy visual navigation
- **Smooth animations** - Professional polish
- **Responsive layout** - Works on all devices

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed
- **AI API Key** (optional - mock mode works without any key!)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd llm_correction_tracker
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements-simple.txt
   ```

   **New dependencies**:
   - `sentence-transformers` - Semantic embeddings
   - `scikit-learn` - PCA projections
   - `numpy` - Vector operations
   - `umap-learn` - UMAP projections

3. **Configure your backend**
   ```bash
   cp .env.example .env
   nano .env  # Edit configuration
   ```

   **Configuration options**:
   ```bash
   # Option 1: Mock Mode (No API key needed!)
   LLM_MODE=mock

   # Option 2: Groq (Recommended - FREE & FAST)
   LLM_MODE=groq
   GROQ_API_KEY=your_key_here  # Get free at https://console.groq.com
   GROQ_MODEL=llama-3.1-8b-instant

   # Option 3: OpenAI (Premium)
   LLM_MODE=openai
   OPENAI_API_KEY=your_key_here
   ```

4. **Start the backend**
   ```bash
   python app.py
   ```
   Backend runs on **http://localhost:5001** (note: changed from 5000)

5. **Open the frontend**
   - **Option A**: Direct file
     ```bash
     cd ../frontend
     open index.html  # or double-click the file
     ```
   - **Option B**: Local server (recommended)
     ```bash
     python -m http.server 8000
     # Visit http://localhost:8000
     ```

## 📖 How to Use

### Step 1: Ask a Question
Type any question. Try these examples:
- "When did the Mexican-American War end?"
- "What's the capital of Australia?"
- "How does photosynthesis work?"

### Step 2: Get AI Response
Click "🚀 Start Learning Session" and get the initial answer!

### Step 3: Provide Corrections
Tell the AI what's wrong or what needs improvement:
- "Actually, the treaty was signed in February 1848"
- "You should mention that Canberra is the capital, not Sydney"
- "Can you explain the light-dependent reactions?"

### Step 4: Explore Semantic Changes! ✨
Once you have 2+ responses, the **semantic analysis** activates:

- **Similarity Heatmap**: Compare all response pairs
- **Vector Movement**: See the semantic trajectory with arrows
- **Sentence Comparison**: Understand which ideas changed
- **Turn-by-Turn Summary**: Plain-English change descriptions

### Step 5: Learn the Theory 🎓
Click sections to expand educational content:
- **Theory boxes** explain concepts
- **Calculation boxes** show formulas
- **Explanation boxes** guide interpretation

### Step 6: Try Interactive Explainers
Click **"🚀 Open Interactive Explainers"** to access:
- **Voting Council** demo (attention mechanism)
- **Probability Funnel** demo (temperature & softmax)

## 🏗️ Project Structure

```
llm_correction_tracker/
├── backend/
│   ├── app.py                    # Flask API + semantic analysis
│   ├── requirements-simple.txt   # Python dependencies
│   ├── .env.example              # Configuration template
│   └── .env                      # Your config (create this!)
├── frontend/
│   ├── index.html                # Main app with visualizations
│   └── explainers.html           # Interactive educational demos
├── old app/                      # Legacy Streamlit version
│   ├── app.py                    # Inspiration source
│   ├── explainers.py             # Original demos
│   └── utils.py                  # Similarity utilities
├── .gitignore
└── README.md
```

## 📊 Visualizations Explained

### 📏 Response Length Chart
Shows how many words the AI uses.
- **Going up?** Adding more detail
- **Going down?** Being more concise

### 😊 Sentiment Analysis
Emotional tone of response.
- **Positive (green)**: Affirming language
- **Negative (red)**: Corrective language

### 🎯 Confidence Level
Based on uncertainty words.
- **Higher**: Definitive statements
- **Lower**: Hedging ("maybe", "perhaps")

### 🔥 Semantic Similarity Heatmap **NEW**
Cosine similarity matrix comparing all responses.
- **Diagonal**: Always 1.0 (self-comparison)
- **Green cells**: High similarity (>0.85)
- **Red cells**: Low similarity (<0.4)
- **What it shows**: Which corrections changed meaning most

### 📍 Vector Movement **NEW**
2D projection of 384-dimensional embeddings.
- **Each point**: One response
- **Arrows**: Path of semantic evolution
- **Color intensity**: Later turns darker
- **What it shows**: AI's "journey" through semantic space

### 📝 Sentence Comparison **NEW**
Heatmap of sentence-to-sentence similarity.
- **Rows**: Previous answer sentences
- **Columns**: Current answer sentences
- **High diagonal**: Core ideas preserved
- **Off-diagonal highs**: Sentence reordering
- **Low values**: Sentences removed/replaced

## 🔧 API Endpoints

### Existing Endpoints

- **POST /api/start-session** - Start conversation
- **POST /api/correct** - Submit correction
- **GET /api/session/:id** - Get session data
- **GET /api/health** - Health check

### **NEW** Semantic Analysis Endpoint

**GET /api/semantic-analysis/:session_id**

Returns comprehensive semantic analysis:
```json
{
  "similarity_matrix": [[1.0, 0.82], [0.82, 1.0]],
  "changes": [{
    "from_turn": 0,
    "to_turn": 1,
    "cosine_similarity": 0.82,
    "angle_degrees": 15.3,
    "jaccard_similarity": 0.65,
    "length_change_pct": 45.2,
    "explanation": "Small adjustment: The answer shifted slightly to fit your correction."
  }],
  "projection_2d": {
    "method": "umap",
    "coordinates": [[0.5, 0.3], [0.6, 0.35]]
  },
  "sentence_comparison": {
    "previous_sentences": ["Sentence 1", "Sentence 2"],
    "current_sentences": ["New sentence 1", "New sentence 2"],
    "similarity_matrix": [[0.95, 0.3], [0.2, 0.88]]
  }
}
```

## 🎨 Customization

### LLM Backend Options

**Mock Mode** (Default):
```python
LLM_MODE=mock
```
- Built-in knowledge: history, geography, science
- No API costs
- Perfect for demos

**Groq** (Recommended):
```python
LLM_MODE=groq
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.1-8b-instant
```
- FREE tier
- Fast inference (~1s)
- [Get key](https://console.groq.com)

**OpenAI** (Premium):
```python
LLM_MODE=openai
OPENAI_API_KEY=your_key
```
- Paid API
- High quality

### Adjusting Semantic Analysis

Edit `backend/app.py`:

```python
# Change embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Larger, more accurate

# Adjust plain-English thresholds
def plain_explanation(cos_sim: float) -> str:
    if cos_sim > 0.98: return "Virtually identical"
    # ... customize thresholds
```

## 📚 Educational Use Cases

### For Teachers
- Demonstrate semantic similarity without jargon
- Show vector representations visually
- Explain attention mechanisms interactively
- Discuss temperature's effect on creativity

### For Students
- **Hands-on learning**: Interact with real AI
- **Visual understanding**: See embeddings as points
- **Concept exploration**: Adjust weights in demos
- **Critical thinking**: Compare similarity metrics

### For Curious Minds
- Understand how chatbots "think"
- Learn about vector embeddings
- Explore attention mechanisms
- See softmax function in action

## 🔬 How It Works Under the Hood

### Semantic Embedding Pipeline
```
Text → sentence-transformers
     ↓
384-dimensional vector
     ↓
Normalized (unit length)
     ↓
Stored with response
```

### Similarity Calculation
```python
cosine_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
# Result: 0.0 (orthogonal) to 1.0 (identical)

angle = arccos(cosine_sim) * 180/π
# Result: 0° (identical) to 180° (opposite)
```

### Dimensionality Reduction
```python
# 384D → 2D for visualization
reducer = umap.UMAP(n_components=2, n_neighbors=5)
coords_2d = reducer.fit_transform(embeddings)
```

## 🤔 Common Questions

**Q: How is this different from the old version?**
A: The old version only tracked word counts. This version uses **semantic embeddings** to measure actual meaning changes, with beautiful heatmaps and 2D projections!

**Q: Do I need an API key?**
A: No! Mock mode works without any API key and has built-in educational responses.

**Q: What's the difference between cosine similarity and Jaccard similarity?**
A:
- **Cosine**: Measures semantic meaning using embeddings (0.0-1.0)
- **Jaccard**: Measures word overlap: |shared words| / |all words| (0.0-1.0)

Example:
- "Paris is France's capital" vs. "The capital of France is Paris"
- Cosine: ~0.95 (very similar meaning)
- Jaccard: ~0.40 (only 40% word overlap)

**Q: What's UMAP vs PCA?**
A:
- **UMAP**: Preserves local structure, better for clusters
- **PCA**: Preserves global variance, simpler math
- App tries UMAP first, falls back to PCA if needed

**Q: Is my data private?**
A:
- **Mock mode**: All local, nothing sent anywhere
- **Groq/OpenAI**: Requests sent to their APIs (see their privacy policies)
- **Sessions**: Stored in memory, cleared on restart

## 🐛 Troubleshooting

### "Semantic analysis unavailable"
- Install missing dependency: `pip install sentence-transformers`
- This downloads ~400MB model on first run (one-time)

### Visualizations not showing
- Check browser console (F12) for errors
- Verify chartjs-chart-matrix plugin loaded
- Try refreshing after backend fully starts

### CORS errors
- Backend must run on port **5001** (not 5000)
- Check `API_URL` in index.html matches your backend
- Use `python -m http.server` instead of `file://`

### Slow embedding computation
- Normal on first run (downloads model)
- Subsequent runs use cached model
- Embeddings computed once per response

## 🚀 Future Enhancements

- [ ] Database persistence (SQLite) for session history
- [ ] Export visualizations as PNG/PDF
- [ ] More interactive demos (beam search, token prediction)
- [ ] Local LLM support (Llama, Mistral)
- [ ] Multi-language embeddings
- [ ] Layer-by-layer token analysis (from old_app)
- [ ] Keyword tracking across semantic space

## 📄 License

MIT License - feel free to use for educational purposes!

## 🙏 Acknowledgments

Built with:
- [sentence-transformers](https://www.sbert.net/) - Semantic embeddings
- [Chart.js](https://www.chartjs.org/) + [Matrix plugin](https://www.chartjs.org/chartjs-chart-matrix/) - Visualizations
- [UMAP](https://umap-learn.readthedocs.io/) - Dimensionality reduction
- [Groq](https://groq.com/) - Fast LLM inference
- Concepts from old_app branch (Streamlit version)

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional similarity metrics
- More interactive demos
- Performance optimizations
- Mobile experience improvements

## 💬 Support

- OpenAI API Docs: https://platform.openai.com/docs
- Groq API Docs: https://console.groq.com/docs
- sentence-transformers: https://www.sbert.net/
- Chart.js: https://www.chartjs.org/docs/

---

**Built with ❤️ to make AI concepts accessible to everyone**

*Transforming abstract mathematics into beautiful, intuitive visualizations* ✨

Happy exploring! 🚀
