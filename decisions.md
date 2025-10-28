# Architecture Decision Records (ADR)

> **Purpose**: Document key architectural and design decisions for the LLM Learning Visualizer project

---

## ADR-001: Multi-Backend LLM Support

**Date**: September 2025
**Status**: ✅ Accepted & Implemented
**Deciders**: Project team

### Context
Initially, the project only supported OpenAI's API. This created barriers:
- Cost for users (OpenAI requires payment)
- Dependency on single vendor
- API key setup complexity for beginners

### Decision
Implement a pluggable LLM backend system with multiple options:
1. **Mock Mode**: Built-in responses, no API needed
2. **Groq**: Free tier, fast inference
3. **OpenAI**: Premium option
4. **Hugging Face**: Attempted but deprecated

### Consequences

**Positive**:
- ✅ Zero-cost demo mode (mock)
- ✅ Free real AI option (Groq)
- ✅ Better resilience to API changes
- ✅ Educational: works anywhere, anytime

**Negative**:
- ❌ More code complexity
- ❌ Different response qualities
- ❌ More testing needed
- ❌ Documentation overhead

### Implementation
```python
LLM_MODE = os.getenv('LLM_MODE', 'mock').lower()

if LLM_MODE == 'mock':
    return mock_llm(messages)
elif LLM_MODE == 'groq':
    return groq_llm(messages)
# ...
```

---

## ADR-002: Semantic Embeddings Over Pure Text Analysis

**Date**: October 2025
**Status**: ✅ Accepted & Implemented
**Deciders**: Project team, informed by old_app insights

### Context
Initial version only tracked surface-level metrics:
- Word count, sentiment, confidence
- No understanding of meaning changes
- Limited educational value

Reviewing the old_app Streamlit version revealed powerful semantic analysis capabilities that users found compelling.

### Decision
Integrate sentence-transformers for semantic embeddings:
- Use all-MiniLM-L6-v2 model (384 dimensions)
- Compute embeddings for all responses
- Calculate cosine similarity, vector angles
- Project to 2D with UMAP/PCA
- Compare at sentence level

### Consequences

**Positive**:
- ✅ Shows MEANING changes, not just text
- ✅ 10x increase in educational value
- ✅ Beautiful visualizations possible
- ✅ Differentiates from competitors

**Negative**:
- ❌ ~400MB model download required
- ❌ Slower first run (model loading)
- ❌ Higher compute requirements
- ❌ Increased dependency size

### Alternatives Considered
1. **Use OpenAI embeddings**: Rejected (requires API key, costs money)
2. **Use simpler TF-IDF**: Rejected (not semantic, just lexical)
3. **No embeddings**: Rejected (insufficient educational value)

### Implementation
```python
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding = embedding_model.encode([text], normalize_embeddings=True)[0]
```

---

## ADR-003: React via CDN (No Build Step)

**Date**: Initial development
**Status**: ✅ Accepted (Reconsidering for future)
**Deciders**: Project team

### Context
Needed a frontend framework but wanted to avoid build complexity for:
- Easy deployment
- Simple development setup
- Low barrier for contributors
- Quick prototyping

### Decision
Use React 18 via CDN with Babel in-browser compilation:
- No webpack/vite configuration
- No npm install for frontend
- Single HTML file deployment
- Babel standalone for JSX

### Consequences

**Positive**:
- ✅ Zero build step
- ✅ Easy to deploy (just HTML files)
- ✅ Fast initial setup
- ✅ Easy for beginners to understand

**Negative**:
- ❌ Slower runtime (Babel compilation)
- ❌ No tree shaking
- ❌ No code splitting
- ❌ Limited TypeScript support
- ❌ Harder to scale

### Future Consideration
May migrate to proper build system when:
- App reaches 100KB+ JavaScript
- Need TypeScript
- Performance becomes issue
- Team grows beyond solo developer

---

## ADR-004: Chart.js Over D3.js

**Date**: Initial development
**Status**: ✅ Accepted
**Deciders**: Project team

### Context
Needed visualization library for charts and heatmaps.

### Decision
Use Chart.js with matrix plugin:
- Simpler API than D3
- Good defaults out of box
- Matrix plugin for heatmaps
- Sufficient for current needs

### Consequences

**Positive**:
- ✅ Fast to implement
- ✅ Beautiful defaults
- ✅ Good documentation
- ✅ Active community

**Negative**:
- ❌ Less flexible than D3
- ❌ Limited for advanced viz
- ❌ Harder to customize deeply
- ❌ May need D3 later for complex features

### Alternatives Considered
1. **D3.js**: Rejected (too complex for current needs)
2. **Plotly**: Rejected (too heavy, over-featured)
3. **Canvas/WebGL**: Rejected (reinventing wheel)

---

## ADR-005: Flask Over FastAPI

**Date**: Initial development
**Status**: ✅ Accepted
**Deciders**: Project team

### Context
Needed Python web framework for REST API.

### Decision
Use Flask 3.0:
- Mature, stable
- Simple for small projects
- Good ecosystem
- Easy CORS handling

### Consequences

**Positive**:
- ✅ Quick to get started
- ✅ Plenty of documentation
- ✅ Lightweight
- ✅ Flask-CORS works great

**Negative**:
- ❌ No automatic API docs
- ❌ No type validation
- ❌ Slower than async frameworks
- ❌ Less modern than FastAPI

### Alternatives Considered
1. **FastAPI**: Rejected (overkill, async not needed)
2. **Django**: Rejected (too heavy for API-only)
3. **Express.js**: Rejected (wanted Python)

### Future Consideration
Might migrate to FastAPI when:
- Need automatic API docs
- WebSocket support required
- Async operations needed
- Type safety becomes priority

---

## ADR-006: In-Memory Sessions Over Database

**Date**: Initial development
**Status**: ⚠️ Temporary (Phase 5 will change)
**Deciders**: Project team

### Context
Needed session storage for conversations.

### Decision
Use Python dict for in-memory storage:
- Fastest development
- No database setup
- Simple for prototyping
- Good for demos

### Consequences

**Positive**:
- ✅ Zero setup overhead
- ✅ Fast access
- ✅ Simple code
- ✅ Good for development

**Negative**:
- ❌ Sessions lost on restart
- ❌ Can't handle many users
- ❌ No history browsing
- ❌ Memory grows unbounded

### Future Change (Phase 5)
Will migrate to SQLite:
- Persistent storage
- Session history
- Better for production
- Reference implementation in old_app

---

## ADR-007: Port 5001 Instead of 5000

**Date**: Mid-development
**Status**: ✅ Accepted
**Deciders**: Project team, informed by user feedback

### Context
Port 5000 commonly in use by:
- AirPlay Receiver on macOS
- Other development servers
- System services

Many users encountered port conflicts.

### Decision
Change default port to 5001:
- Less commonly used
- Still in common dev range
- Easy to remember (5000 + 1)

### Consequences

**Positive**:
- ✅ Fewer port conflicts
- ✅ Better user experience
- ✅ Less support burden

**Negative**:
- ❌ Breaking change for existing users
- ❌ Documentation updates needed
- ❌ Not standard (5000 is convention)

### Implementation
```python
PORT = int(os.getenv('PORT', 5001))  # Changed from 5000
```

And in frontend:
```javascript
const API_URL = 'http://localhost:5001/api';  // Changed from 5000
```

---

## ADR-008: Collapsible Sections for Content Organization

**Date**: October 2025
**Status**: ✅ Accepted & Implemented
**Deciders**: Project team, user feedback

### Context
As educational content grew, pages became overwhelming:
- Too much information at once
- Cognitive overload for users
- Hard to find specific sections

### Decision
Implement collapsible sections with:
- Click-to-expand headers
- Color-coded by content type
- Icons for visual recognition
- Smooth animations
- Default states (important ones open)

### Consequences

**Positive**:
- ✅ Reduced cognitive load
- ✅ Progressive disclosure
- ✅ Better mobile experience
- ✅ Professional appearance

**Negative**:
- ❌ Extra clicks required
- ❌ More state management
- ❌ Slightly more complex code

### User Feedback
Overwhelmingly positive - users love being able to control information density.

---

## ADR-009: UMAP with PCA Fallback

**Date**: October 2025
**Status**: ✅ Accepted & Implemented
**Deciders**: Project team

### Context
Needed 2D projection of 384D embeddings for visualization.

### Decision
Try UMAP first, fallback to PCA:
- UMAP better preserves local structure
- UMAP requires minimum neighbors
- PCA always works
- PCA simpler, more predictable

### Consequences

**Positive**:
- ✅ Best visualization when possible
- ✅ Reliable fallback
- ✅ Graceful degradation

**Negative**:
- ❌ UMAP can be slow
- ❌ Two dependencies
- ❌ Non-deterministic behavior

### Implementation
```python
try:
    import umap
    reducer = umap.UMAP(n_components=2, ...)
    return reducer.fit_transform(X)
except:
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(X)
```

---

## ADR-010: Three Content Box Types

**Date**: October 2025
**Status**: ✅ Accepted & Implemented
**Deciders**: Project team

### Context
Educational content needs clear organization.

### Decision
Create three distinct box types:
1. **Theory Boxes** (Cyan): Conceptual explanations
2. **Explanation Boxes** (Purple): How to interpret visualizations
3. **Calculation Boxes** (Orange): Mathematical formulas

### Consequences

**Positive**:
- ✅ Clear visual hierarchy
- ✅ Users know what to expect
- ✅ Scannable content
- ✅ Color-coded for quick reference

**Negative**:
- ❌ Design consistency burden
- ❌ More CSS to maintain

### User Feedback
Users quickly learn the color coding and appreciate the structure.

---

## ADR-011: Mock Mode as Default

**Date**: October 2025
**Status**: ✅ Accepted
**Deciders**: Project team

### Context
New users struggled with API key setup.

### Decision
Make mock mode the default:
- Works immediately
- No configuration needed
- Built-in educational responses
- Perfect for demos

### Consequences

**Positive**:
- ✅ Instant usability
- ✅ No setup friction
- ✅ Reliable for demos
- ✅ Works offline

**Negative**:
- ❌ Limited response variety
- ❌ Not "real" AI
- ❌ Some might not realize other modes exist

### Mitigation
- Clear documentation of other modes
- Health endpoint shows current mode
- Educational responses are high quality

---

## ADR-012: Plain-English Similarity Descriptions

**Date**: October 2025
**Status**: ✅ Accepted & Implemented
**Deciders**: Project team

### Context
Raw similarity scores (0.82) meaningless to non-technical users.

### Decision
Create interpretive text based on similarity ranges:
- > 0.95: "Tiny tweak"
- > 0.85: "Small adjustment"
- > 0.70: "Moderate change"
- > 0.40: "Big change"
- < 0.40: "Major rethink"

### Consequences

**Positive**:
- ✅ Instantly understandable
- ✅ Memorable categories
- ✅ Sharable insights
- ✅ Helps users learn scale

**Negative**:
- ❌ Somewhat arbitrary thresholds
- ❌ Might oversimplify
- ❌ Context-dependent accuracy

### Implementation
```python
def plain_explanation(cos_sim: float) -> str:
    if cos_sim > 0.95: return "Tiny tweak: ..."
    # ...
```

---

## ADR-013: Sentence-Level Comparison

**Date**: October 2025
**Status**: ✅ Accepted & Implemented
**Deciders**: Project team, inspired by old_app

### Context
Document-level similarity doesn't show which specific ideas changed.

### Decision
Compute embeddings for each sentence and compare:
- Split responses into sentences
- Embed each separately
- Create sentence-to-sentence similarity matrix
- Visualize as heatmap

### Consequences

**Positive**:
- ✅ Reveals structural changes
- ✅ Shows idea evolution
- ✅ More granular understanding
- ✅ Unique feature

**Negative**:
- ❌ More embeddings to compute
- ❌ Complex visualization
- ❌ Can be noisy with short sentences

---

## Questions for Future ADRs

### Under Consideration
1. **Database choice**: SQLite vs. PostgreSQL vs. Cloud?
2. **Authentication**: Roll our own vs. Auth0 vs. Firebase?
3. **Hosting**: Vercel vs. Railway vs. AWS vs. self-hosted?
4. **Frontend framework**: Stay React or try Svelte/Vue?
5. **Type system**: Add TypeScript? Python type checking?
6. **Testing strategy**: Unit vs. Integration focus?
7. **Monitoring**: Sentry vs. LogRocket vs. custom?

### Need More Data For
- WebSocket for real-time features?
- GraphQL instead of REST?
- Containerization (Docker) strategy?
- CI/CD pipeline details?
- Error boundary strategy?
- Internationalization approach?

---

## Decision-Making Process

### Criteria for Good Decisions
1. **User Value**: Does it help users learn?
2. **Maintainability**: Can we support it long-term?
3. **Simplicity**: Is it the simplest solution?
4. **Performance**: Is it fast enough?
5. **Cost**: Within budget constraints?

### When to Create an ADR
- Significant architectural choice
- Technology selection
- API design pattern
- Data model change
- Performance tradeoff
- Security decision

### ADR Format
- **Context**: What's the situation?
- **Decision**: What did we decide?
- **Consequences**: What are the impacts?
- **Alternatives**: What else was considered?

---

*Last Updated: October 27, 2025*
