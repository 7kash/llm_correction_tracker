# CLAUDE.md - AI Assistant Guide

## Project Overview

**LLM Learning Visualizer** is an educational web application that demonstrates how Large Language Models adapt and change their responses when corrected. The project uses semantic similarity analysis with beautiful visualizations to make AI concepts accessible to non-technical users.

## Core Purpose

Help non-technical users understand:
- How LLMs process corrections
- Semantic vs. lexical similarity
- Vector embeddings and meaning representation
- Attention mechanisms and temperature effects

## Technology Stack

### Backend (Flask + Python)
- **Flask 3.0.0**: REST API server
- **sentence-transformers**: 384D semantic embeddings (all-MiniLM-L6-v2)
- **scikit-learn**: PCA dimensionality reduction
- **umap-learn**: UMAP projections for visualization
- **numpy**: Vector operations
- **flask-cors**: Cross-origin support

### Frontend (React + Vanilla JS)
- **React 18**: UI framework (via CDN, no build step)
- **Chart.js 4.4**: Visualization library
- **chartjs-chart-matrix**: Heatmap plugin
- **Tailwind CSS**: Styling (via CDN)

### LLM Backends (Multiple Options)
- **Mock Mode** (default): No API key needed, built-in responses
- **Groq**: FREE tier, fast inference (llama-3.1-8b-instant)
- **Hugging Face**: Deprecated/restricted free API
- **OpenAI**: Premium paid API

## Project Structure

```
llm_correction_tracker/
├── backend/
│   ├── app.py                    # Main Flask application
│   ├── requirements-simple.txt   # Python dependencies
│   ├── .env.example              # Configuration template
│   └── .env                      # Local configuration (not in git)
├── frontend/
│   ├── index.html                # Main app with semantic visualizations
│   └── explainers.html           # Interactive educational demos
├── old app/                      # Legacy Streamlit version (reference)
├── README.md                     # User-facing documentation
├── DEMO_GUIDE.md                 # Demo instructions
└── Documentation/ (this file and others)
```

## Key Features Implemented

### 1. Semantic Similarity Analysis
- **Embeddings**: 384D vectors via sentence-transformers
- **Cosine Similarity**: Measures semantic meaning (0.0-1.0)
- **Jaccard Similarity**: Word overlap percentage
- **Vector Angles**: Geometric difference in degrees
- **Plain-English**: "Tiny tweak" to "Major rethink" descriptions

### 2. Advanced Visualizations
- **Similarity Heatmap**: Matrix showing cosine similarity between all responses
- **Vector Movement**: 2D projection (UMAP/PCA) with arrows showing semantic path
- **Sentence Comparison**: Heatmap comparing individual sentences
- **Basic Analytics**: Word count, sentiment, confidence charts

### 3. Interactive Explainers
- **Voting Council**: Attention mechanism demo with adjustable recency/relevance
- **Probability Funnel**: Temperature and softmax visualization

### 4. Educational Content
- **Theory Boxes** (cyan): Conceptual explanations
- **Explanation Boxes** (purple): Visualization interpretation
- **Calculation Boxes** (orange): Mathematical formulas with examples

## API Endpoints

### Core Endpoints
- `POST /api/start-session`: Initialize conversation with question
- `POST /api/correct`: Submit correction and get updated response
- `GET /api/session/<id>`: Retrieve full session data with analytics
- `GET /api/semantic-analysis/<id>`: Get comprehensive semantic analysis
- `GET /api/health`: Health check and mode information

### Semantic Analysis Response Format
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
    "explanation": "Small adjustment: ..."
  }],
  "projection_2d": {
    "method": "umap",
    "coordinates": [[0.5, 0.3], [0.6, 0.35]]
  },
  "sentence_comparison": {
    "previous_sentences": ["..."],
    "current_sentences": ["..."],
    "similarity_matrix": [[0.95, 0.3], [0.2, 0.88]]
  }
}
```

## Development Guidelines

### When Making Changes

1. **Backend Changes**:
   - Keep semantic analysis separate from basic text analysis
   - Maintain backward compatibility for sessions
   - Test with all LLM modes (mock, groq, openai)
   - Document any new environment variables in .env.example

2. **Frontend Changes**:
   - Preserve collapsible section structure
   - Maintain educational content (Theory/Explanation/Calculation boxes)
   - Test visualizations with 2-10 responses
   - Ensure mobile responsiveness

3. **Adding Features**:
   - Consider non-technical users (avoid jargon)
   - Add Theory box explaining the concept
   - Include worked examples in Calculation boxes
   - Provide plain-English interpretations

### Code Quality Standards

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Use const/let, avoid var
- **Comments**: Explain WHY, not WHAT
- **Functions**: Keep small, single responsibility
- **Error Handling**: Graceful degradation (e.g., UMAP → PCA fallback)

### Testing Checklist

- [ ] Mock mode works without any API keys
- [ ] Groq mode works with valid API key
- [ ] Semantic analysis loads after 2+ responses
- [ ] All visualizations render correctly
- [ ] Explainers page works standalone
- [ ] Mobile/tablet layouts work
- [ ] Browser cache doesn't break updates (hard refresh test)

## Common Issues & Solutions

### Backend Won't Start
- Check Python 3.8+ installed
- Install dependencies: `pip install -r requirements-simple.txt`
- First run downloads sentence-transformers model (~400MB)
- Port conflicts: Use PORT=5001 or higher

### Frontend Not Loading
- Hard refresh browser (Cmd+Shift+R / Ctrl+Shift+R)
- Check API_URL in index.html matches backend port
- Use local HTTP server, not file:// protocol
- Check browser console for CORS errors

### Semantic Analysis Not Appearing
- Need at least 2 responses (initial + 1 correction)
- Check if sentence-transformers installed
- Look for errors in backend console
- Verify /api/semantic-analysis/<id> returns 200

### Visualizations Look Wrong
- Verify chartjs-chart-matrix plugin loaded
- Check Chart.js version is 4.4+
- Test with 2-3 responses first
- Clear browser cache and reload

## Recent Major Changes

### Latest (October 2025)
- ✅ Added comprehensive semantic similarity analysis
- ✅ Implemented similarity heatmaps and vector movement plots
- ✅ Created sentence-level comparison heatmaps
- ✅ Built interactive explainers (Voting Council, Probability Funnel)
- ✅ Added extensive educational content with theory boxes
- ✅ Implemented collapsible sections for better UX
- ✅ Updated to Groq llama-3.1-8b-instant (previous model decommissioned)

### Historical
- Added Groq backend (free, fast alternative to OpenAI)
- Created mock mode with built-in educational responses
- Fixed port conflicts (5000 → 5001)
- Deprecated Hugging Face support (API restrictions)

## Future Enhancement Ideas

### High Priority
- [ ] Database persistence (SQLite for session history)
- [ ] Export visualizations as PNG/PDF
- [ ] More interactive demos (beam search, token prediction)

### Medium Priority
- [ ] Local LLM support (Llama, Mistral via Ollama)
- [ ] Multi-language embeddings
- [ ] Layer-by-layer token analysis (from old_app)
- [ ] Keyword tracking across semantic space

### Low Priority
- [ ] User accounts and saved sessions
- [ ] Side-by-side model comparison
- [ ] Real-time collaboration features
- [ ] Mobile app version

## Important Notes for AI Assistants

### When Helping Users

1. **Always check LLM_MODE** in .env - affects behavior significantly
2. **Port issues are common** - default changed from 5000 to 5001
3. **First-time setup is slow** - sentence-transformers downloads ~400MB model
4. **Browser cache causes confusion** - teach users to hard refresh
5. **Mock mode is best for demos** - no API keys needed

### Code Organization Patterns

- **Mock LLM**: Lines 96-230 in app.py (knowledge base for demos)
- **Semantic Analysis**: Lines 69-127 (embedding and similarity functions)
- **Semantic Endpoint**: Lines 613-736 (comprehensive analysis)
- **Frontend API URL**: Line 132 in index.html
- **Visualization Updates**: Lines 462-706 in index.html

### Don't Break These

- Collapsible section structure (users love it)
- Educational content organization (Theory/Explanation/Calculation)
- Mock mode knowledge base (used in demos)
- Plain-English similarity descriptions (key user-facing feature)
- Port 5001 default (many users already configured)

## Getting Help

- Check README.md for user documentation
- See DEMO_GUIDE.md for presentation tips
- Review old_app/ branch for original Streamlit implementation
- Git history shows evolution of semantic analysis features

## Quick Reference Commands

```bash
# Start backend
cd backend && python app.py

# Start frontend
cd frontend && python -m http.server 8000

# Install dependencies
pip install -r backend/requirements-simple.txt

# Check what's using a port
lsof -i :5001

# Test backend health
curl http://localhost:5001/api/health

# View git history
git log --oneline --graph

# Check current branch
git branch
```

---

**Remember**: This is an educational tool. Keep it accessible, visual, and jargon-free!
