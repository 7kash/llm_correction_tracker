# Architecture

## System Overview

```
┌─────────────┐      HTTP      ┌──────────────┐
│   Browser   │ ◄────────────► │  Flask API   │
│  (React 18) │   REST/JSON    │  (Port 5001) │
└─────────────┘                └──────┬───────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                    ┌────────┐  ┌─────────┐  ┌─────────┐
                    │  Mock  │  │  Groq   │  │ OpenAI  │
                    │  LLM   │  │   API   │  │   API   │
                    └────────┘  └─────────┘  └─────────┘
```

## Core Components

### Backend (Flask)
**File**: `backend/app.py` (~800 lines)

**Key Modules**:
- **LLM Router** (lines 96-230): Mock/Groq/OpenAI mode switching
- **Semantic Engine** (lines 69-127): Embeddings, similarity, projections
- **Session Manager** (lines 50-67): In-memory dict storage
- **REST API** (lines 250-736): 5 endpoints

**Data Flow**:
1. Load sentence-transformers model on startup (384D embeddings)
2. Receive question → Call LLM → Generate response
3. Compute embedding → Store with response
4. On correction: Generate new response → Compute similarity
5. Return analytics + semantic analysis

### Frontend (React via CDN)
**File**: `frontend/index.html` (~1200 lines)

**Structure**:
- **State Management** (line 140): Single `sessionData` object
- **API Client** (lines 180-340): Fetch wrapper functions
- **Visualizations** (lines 462-706): Chart.js integration
- **Collapsible Sections** (lines 850-1100): Educational content

**Libraries**:
- React 18.2.0 (CDN)
- Chart.js 4.4.0 + Matrix plugin
- Tailwind CSS 3.4.0
- Babel Standalone (JSX compilation)

### Explainers Page
**File**: `frontend/explainers.html` (~600 lines)

Standalone demos:
- Voting Council (attention mechanism)
- Probability Funnel (temperature/softmax)

## API Design

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/start-session` | Initialize conversation |
| POST | `/api/correct` | Submit correction |
| GET | `/api/session/<id>` | Retrieve session + basic analytics |
| GET | `/api/semantic-analysis/<id>` | Get semantic similarity data |
| GET | `/api/health` | Health check + mode info |

### Data Models

**Session Object**:
```python
{
    "session_id": str,
    "question": str,
    "responses": [
        {
            "turn": int,
            "correction": str,
            "response": str,
            "timestamp": float,
            "embedding": np.array(384),  # Not sent to client
            "word_count": int,
            "sentiment": float,
            "confidence": float
        }
    ]
}
```

**Semantic Analysis Response**:
```python
{
    "similarity_matrix": float[][],      # N×N cosine similarities
    "changes": [{                        # Turn-by-turn deltas
        "from_turn": int,
        "to_turn": int,
        "cosine_similarity": float,
        "angle_degrees": float,
        "jaccard_similarity": float,
        "length_change_pct": float,
        "explanation": str
    }],
    "projection_2d": {
        "method": "umap" | "pca",
        "coordinates": float[][]
    },
    "sentence_comparison": {
        "previous_sentences": str[],
        "current_sentences": str[],
        "similarity_matrix": float[][]
    }
}
```

## Technology Stack

### Backend
- **Flask 3.0.0**: Web framework
- **sentence-transformers 2.2.2**: all-MiniLM-L6-v2 (384D)
- **scikit-learn 1.3.0**: PCA fallback
- **umap-learn 0.5.4**: Primary dimensionality reduction
- **numpy 1.24.3**: Vector operations
- **requests 2.31.0**: HTTP client for LLM APIs

### Frontend
- **React 18.2.0**: UI framework (CDN, no build)
- **Chart.js 4.4.0**: Visualizations
- **chartjs-chart-matrix**: Heatmap plugin
- **Tailwind CSS 3.4.0**: Styling
- **Babel Standalone**: JSX → JS

### LLM Backends
- **Mock**: Built-in responses (lines 96-230)
- **Groq**: llama-3.1-8b-instant (fast, free)
- **OpenAI**: gpt-3.5-turbo (paid)

## Deployment

### Development
```bash
# Terminal 1: Backend
cd backend && python app.py  # Port 5001

# Terminal 2: Frontend
cd frontend && python -m http.server 8000
```

### Configuration
**File**: `backend/.env`
```bash
LLM_MODE=mock|groq|openai
GROQ_API_KEY=...
OPENAI_API_KEY=...
PORT=5001
```

### First-Time Setup
1. Install dependencies: `pip install -r requirements-simple.txt`
2. Downloads sentence-transformers model (~400MB) on first run
3. Model cached at `~/.cache/torch/sentence_transformers/`

## Performance

### Latency
- **Mock mode**: <10ms (no network)
- **Groq**: ~1-2s (fast inference)
- **OpenAI**: ~2-4s (standard)
- **Embeddings**: ~50-100ms per response (local)
- **UMAP projection**: ~200-500ms (depends on # responses)

### Memory
- **Sentence-transformers model**: ~400MB RAM
- **Session storage**: ~5KB per response
- **No persistence**: Cleared on restart

## Security

### Current State
- ⚠️ No authentication
- ⚠️ No rate limiting
- ⚠️ Sessions in memory (not encrypted)
- ✅ CORS enabled (flask-cors)
- ✅ API keys in .env (not committed)

### Recommendations (Future)
- Add rate limiting (Flask-Limiter)
- Implement session expiration
- Add HTTPS in production
- Sanitize user inputs
- Move to database with encryption

## Scalability Limitations

**Current bottlenecks**:
1. In-memory sessions → Lost on restart
2. No horizontal scaling (shared state)
3. UMAP gets slow with >50 responses
4. Synchronous LLM calls (blocks thread)

**Phase 5 improvements** (see plan.md):
- SQLite persistence
- Background job queue for embeddings
- Response caching

## Code Organization

### Backend Structure
```python
# app.py structure
- Global config (lines 1-50)
- Embedding model initialization (51-68)
- Semantic analysis functions (69-127)
- Mock LLM knowledge base (96-230)
- Session management (231-249)
- API routes (250-736)
```

### Frontend Structure
```javascript
// index.html structure
- CDN imports (lines 1-50)
- React component definition (140-1150)
  - State management
  - API client methods
  - Event handlers
  - Render method with sections
- Chart.js helpers (462-706)
- Initialization (1151-1200)
```

## Key Design Patterns

### Backend
- **Strategy Pattern**: LLM mode switching
- **Singleton**: Embedding model (global)
- **Factory**: Session creation
- **Adapter**: LLM API wrappers

### Frontend
- **Container/Presentation**: React component split
- **Observer**: State updates trigger re-renders
- **Lazy Loading**: Semantic analysis on demand
- **Progressive Enhancement**: Collapsible sections

## Error Handling

### Backend
```python
try:
    # Attempt UMAP
    reducer = umap.UMAP(...)
except:
    # Fallback to PCA
    reducer = PCA(...)
```

### Frontend
```javascript
try {
    const data = await fetchSemanticAnalysis();
} catch (error) {
    console.error("Semantic analysis failed:", error);
    // Continue without semantic viz
}
```

**Philosophy**: Graceful degradation, never block core functionality

## Testing Strategy

### Current Testing
- Manual testing only
- No unit tests
- No integration tests

### Recommended (Phase 6)
- Unit tests: Semantic similarity functions
- Integration tests: API endpoints
- E2E tests: Cypress/Playwright for frontend
- Mock LLM responses for deterministic tests

## Monitoring

### Current
- Console logs only
- No structured logging
- No metrics collection

### Future (Phase 7)
- Structured logging (JSON)
- API latency tracking
- Error rate monitoring
- LLM token usage tracking

---

**Last Updated**: 2025-10-28
**See Also**: CLAUDE.md, plan.md, decisions.md
