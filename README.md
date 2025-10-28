# 🧠 LLM Inference Tracker

⚠️ **NEW VERSION AVAILABLE**: [Gradio App](gradio_app/) with **real model internals** (attention, layer trajectories, logit lens)

**Track how Large Language Models change their responses when corrected** — with visualizations of internal mechanisms!

![Status](https://img.shields.io/badge/Status-Migrating-orange) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Gradio](https://img.shields.io/badge/Gradio-4.19-blue) ![AI](https://img.shields.io/badge/AI-Model_Internals-purple)

📚 **[Full Documentation](CLAUDE.md)** | 🏗️ **[Architecture](architecture.md)** | 📋 **[Roadmap](plan.md)** | ✅ **[Tasks](todo.md)**

## 🚀 Quick Start (New Gradio Version)

```bash
cd gradio_app
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

See [gradio_app/README.md](gradio_app/README.md) for details.

---

## ⚠️ Legacy Version Below

The documentation below describes the **old Flask+React version** with semantic similarity analysis. This version is being replaced by the **Gradio app** (see above) which uses **real model internals** instead of text similarity approximations.

**Use the Gradio app** for:
- Real attention weights from the model
- Layer-by-layer hidden state tracking
- Logit lens (what model "thinks" at each layer)

**Old version features** (semantic similarity):
- Cosine similarity heatmaps
- Vector movement in 2D (UMAP/PCA)
- Sentence-level comparison

---

## ✨ What Was New (Legacy)

## 🎯 What It Does

1. **Ask** the AI a question (OpenAI, Groq, or mock mode)
2. **Correct** when it's wrong
3. **Visualize** how responses change:
   - Semantic similarity (cosine, Jaccard, angles)
   - 2D semantic space evolution
   - Sentence-level comparisons
   - Length, sentiment, confidence trends

## 🚀 Key Features

### Semantic Analysis 🔥
- **384D embeddings** via sentence-transformers
- **Cosine similarity** with plain-English descriptions (Tiny tweak → Major rethink)
- **Vector movement** in 2D space (UMAP/PCA)
- **Sentence comparison** heatmaps

### Similarity Metrics 📐
- **Cosine**: Semantic meaning (0.0-1.0)
- **Jaccard**: Word overlap %
- **Vector Angle**: Geometric difference
- **Length Change**: Response size Δ

### Interactive Demos 🎓
- **Voting Council**: Attention mechanism with adjustable weights
- **Probability Funnel**: Temperature & softmax visualization
- Access via `explainers.html`

### Educational Content 📚
- **Theory boxes**: Conceptual explanations
- **Calculation boxes**: Formulas with examples
- **Explanation boxes**: Interpretation guides

### Design 🎨
- Collapsible sections, color-coded themes
- Smooth animations, responsive layout
- See [architecture.md](architecture.md) for technical details

## 🚀 Quick Start

### Install & Run

```bash
# 1. Install dependencies (Python 3.8+)
cd backend
pip install -r requirements-simple.txt

# 2. Configure (optional - mock mode works without API key)
cp .env.example .env
# Edit .env: Set LLM_MODE=mock|groq|openai

# 3. Start backend
python app.py  # Runs on port 5001

# 4. Start frontend (new terminal)
cd ../frontend
python -m http.server 8000  # Visit http://localhost:8000
```

### Configuration Options

```bash
# Mock Mode (default - no API key needed)
LLM_MODE=mock

# Groq (FREE & fast)
LLM_MODE=groq
GROQ_API_KEY=your_key  # Get at https://console.groq.com

# OpenAI (paid)
LLM_MODE=openai
OPENAI_API_KEY=your_key
```

**First run**: Downloads sentence-transformers model (~400MB, one-time)

## 📖 How to Use

1. **Ask a question**: "When did the Mexican-American War end?"
2. **Get response**: Click "🚀 Start Learning Session"
3. **Provide correction**: "The treaty was signed in February 1848"
4. **Explore visualizations**: After 2+ responses, semantic analysis activates
   - Similarity heatmap (all response pairs)
   - Vector movement (2D trajectory)
   - Sentence comparison (which ideas changed)
5. **Learn theory**: Expand collapsible sections for educational content
6. **Try demos**: Click "🚀 Open Interactive Explainers"

## 🏗️ Project Structure

```
llm_correction_tracker/
├── backend/
│   ├── app.py                    # Flask API + semantic analysis
│   ├── requirements-simple.txt   # Dependencies
│   └── .env                      # Configuration (create from .env.example)
├── frontend/
│   ├── index.html                # Main app
│   └── explainers.html           # Interactive demos
├── CLAUDE.md                     # AI assistant guide
├── architecture.md               # Technical details
├── plan.md                       # Roadmap
└── README.md                     # This file
```

See [architecture.md](architecture.md) for API endpoints, data models, and technical details.

## 🤔 FAQ

**Do I need an API key?**
No! Mock mode works without any API key.

**Cosine vs Jaccard similarity?**
- **Cosine**: Semantic meaning via embeddings (0.0-1.0)
- **Jaccard**: Word overlap % (0.0-1.0)

**UMAP vs PCA?**
App tries UMAP first (better clusters), falls back to PCA if needed.

**Is data private?**
- Mock: All local
- Groq/OpenAI: Sent to their APIs
- Sessions: In-memory, cleared on restart

**Customization?**
Edit `backend/app.py` to change embedding models or thresholds. See [CLAUDE.md](CLAUDE.md) for details.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Semantic analysis unavailable" | `pip install sentence-transformers` (~400MB download) |
| Visualizations not showing | Check browser console (F12), refresh after backend starts |
| CORS errors | Backend must run on port 5001, use `python -m http.server` |
| Slow first run | Normal - downloads model once, cached afterward |

See [CLAUDE.md](CLAUDE.md) for detailed troubleshooting guide.

## 🚀 Future Plans

See [plan.md](plan.md) for full roadmap. Next phases:
- **Phase 5**: SQLite persistence, export visualizations
- **Phase 6**: More demos (beam search, token prediction)
- **Phase 7**: Local LLM support (Ollama)
- **Phase 8**: Multi-language embeddings

## 📄 License & Contributing

**MIT License** - Free for educational use!

Contributions welcome! See [todo.md](todo.md) for task list and priorities.

## 🙏 Built With

[sentence-transformers](https://www.sbert.net/) • [Chart.js](https://www.chartjs.org/) • [UMAP](https://umap-learn.readthedocs.io/) • [Groq](https://groq.com/)

---

**Making AI concepts accessible through beautiful visualizations** ✨
