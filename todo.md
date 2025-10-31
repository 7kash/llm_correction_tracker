# TODO - LLM Learning Visualizer

> **Last Updated**: October 27, 2025
> **Next Review**: November 2025

---

## 🔥 Immediate / Critical

### Bug Fixes
- [ ] **Fix backend connection issues on Mac Safari**
  - Issue: Users getting "Load failed" errors
  - Root cause: Port conflicts and CORS
  - Solution: Verify .env PORT configuration, test CORS headers
  - Priority: P0 - Blocking demo usage

- [ ] **Resolve port conflicts documentation**
  - Update all docs to consistently reference port 5001
  - Add troubleshooting section for port conflicts
  - Create startup script that handles port detection
  - Priority: P1 - Reduces support burden

### Documentation
- [x] Create CLAUDE.md for AI assistants
- [x] Create plan.md with project roadmap
- [x] Create todo.md (this file)
- [ ] Create decisions.md for architecture decisions
- [ ] Create architecture.md for technical details
- [ ] Add CHANGELOG.md for version history
- [ ] Create CONTRIBUTING.md for potential contributors

---

## 📊 Phase 5: Persistence & Export (Next Sprint)

### Database Integration
- [ ] **Implement SQLite database**
  - [ ] Copy db.py structure from old_app
  - [ ] Create database schema (sessions, turns, embeddings)
  - [ ] Add migration support
  - [ ] Test with 100+ sessions
  - Priority: P1
  - Estimated: 1 week

- [ ] **Session management UI**
  - [ ] Add "Save Session" button
  - [ ] Create session browser/history
  - [ ] Implement "Load Session" functionality
  - [ ] Add session deletion
  - Priority: P1
  - Estimated: 3 days

- [ ] **Session persistence testing**
  - [ ] Test server restart doesn't lose data
  - [ ] Verify concurrent sessions work
  - [ ] Load testing with 50+ sessions
  - Priority: P1
  - Estimated: 2 days

### Export Features
- [ ] **Chart export to PNG**
  - [ ] Add "Download Chart" buttons
  - [ ] Implement Chart.js toBase64Image()
  - [ ] Handle all chart types (heatmap, scatter, bar, line)
  - Priority: P2
  - Estimated: 2 days

- [ ] **Session export to JSON**
  - [ ] Create export endpoint
  - [ ] Include all data (responses, embeddings, analytics)
  - [ ] Add import functionality
  - Priority: P2
  - Estimated: 1 day

- [ ] **PDF Report generation**
  - [ ] Research PDF library (ReportLab? WeasyPrint?)
  - [ ] Design PDF template
  - [ ] Include all visualizations
  - [ ] Add summary statistics
  - Priority: P3
  - Estimated: 1 week

---

## 🎓 Phase 6: Advanced Demos (Future)

### Token Prediction Demo
- [ ] Design UI mockup
- [ ] Implement backend token probability endpoint
- [ ] Create interactive probability adjuster
- [ ] Add beam search visualization
- [ ] Write educational content
- Priority: P2
- Estimated: 2 weeks

### Layer-by-Layer Analysis
- [ ] Port code from old_app/layerwise_next_token.py
- [ ] Adapt for current architecture
- [ ] Create layer selection UI
- [ ] Add attention heatmap per layer
- [ ] Optimize performance (layers are compute-heavy)
- Priority: P2
- Estimated: 2 weeks

### Context Window Demo
- [ ] Design visualization for context window
- [ ] Show what fits vs. what's truncated
- [ ] Visualize attention across context
- [ ] Add educational explanation
- Priority: P3
- Estimated: 1 week

---

## 🎨 UX/UI Improvements

### Mobile Experience
- [ ] **Test on actual mobile devices**
  - [ ] iPhone Safari
  - [ ] Android Chrome
  - [ ] iPad
  - Priority: P1

- [ ] **Responsive layout fixes**
  - [ ] Heatmaps too small on mobile
  - [ ] Theory boxes text size
  - [ ] Chart legends overlapping
  - Priority: P1
  - Estimated: 3 days

- [ ] **Touch interactions**
  - [ ] Add touch-friendly controls
  - [ ] Improve slider usability
  - [ ] Test chart zoom/pan
  - Priority: P2
  - Estimated: 2 days

### Visual Polish
- [ ] **Loading states**
  - [ ] Add spinners for slow operations
  - [ ] Show progress for embedding computation
  - [ ] Skeleton screens for charts
  - Priority: P2
  - Estimated: 2 days

- [ ] **Error messages**
  - [ ] Better error UI (not just alerts)
  - [ ] Specific troubleshooting steps
  - [ ] Inline validation
  - Priority: P2
  - Estimated: 1 day

- [ ] **Onboarding tour**
  - [ ] First-time user guide
  - [ ] Highlight key features
  - [ ] Interactive walkthrough
  - Priority: P3
  - Estimated: 1 week

---

## 🔧 Technical Debt

### Backend Refactoring
- [ ] **Split app.py into modules**
  - [ ] routes.py for endpoints
  - [ ] llm.py for LLM integrations
  - [ ] semantic.py for similarity analysis
  - [ ] utils.py for helpers
  - Priority: P2
  - Estimated: 2 days

- [ ] **Add API documentation**
  - [ ] OpenAPI/Swagger spec
  - [ ] Interactive API docs
  - [ ] Example requests/responses
  - Priority: P2
  - Estimated: 1 day

- [ ] **Error handling improvements**
  - [ ] Standardized error responses
  - [ ] Logging to file
  - [ ] Error monitoring (Sentry?)
  - Priority: P2
  - Estimated: 2 days

### Frontend Refactoring
- [ ] **State management**
  - [ ] Centralize state (React Context or Redux)
  - [ ] Reduce prop drilling
  - [ ] Better session state handling
  - Priority: P3
  - Estimated: 1 week

- [ ] **Component extraction**
  - [ ] Extract reusable chart components
  - [ ] Separate educational content components
  - [ ] Create component library
  - Priority: P3
  - Estimated: 3 days

- [ ] **Build process**
  - [ ] Move from CDN to bundled React
  - [ ] Add TypeScript
  - [ ] Webpack/Vite configuration
  - [ ] Tree shaking for smaller bundle
  - Priority: P3
  - Estimated: 1 week

### Testing
- [ ] **Backend unit tests**
  - [ ] Test similarity calculations
  - [ ] Test LLM mode switching
  - [ ] Test API endpoints
  - [ ] Coverage > 70%
  - Priority: P2
  - Estimated: 1 week

- [ ] **Frontend tests**
  - [ ] Component tests (React Testing Library)
  - [ ] Integration tests
  - [ ] E2E tests (Playwright/Cypress)
  - Priority: P3
  - Estimated: 1 week

- [ ] **Performance testing**
  - [ ] Load test with many sessions
  - [ ] Benchmark embedding computation
  - [ ] Memory leak testing
  - Priority: P2
  - Estimated: 3 days

---

## 📚 Content & Documentation

### Educational Content
- [ ] **Add more theory sections**
  - [ ] Transformer architecture
  - [ ] Self-attention vs. cross-attention
  - [ ] Positional encodings
  - [ ] Layer normalization
  - Priority: P2

- [ ] **Create video tutorials**
  - [ ] Basic usage walkthrough
  - [ ] Understanding semantic similarity
  - [ ] Reading the visualizations
  - [ ] Explainer demos deep dive
  - Priority: P3

- [ ] **Write blog posts**
  - [ ] "How semantic similarity works"
  - [ ] "Building an LLM educator"
  - [ ] "From word counts to embeddings"
  - Priority: P3

### Documentation Improvements
- [ ] **API documentation**
  - [ ] Complete endpoint descriptions
  - [ ] Request/response examples
  - [ ] Error codes and meanings
  - Priority: P2

- [ ] **Deployment guide**
  - [ ] Production setup instructions
  - [ ] Environment configuration
  - [ ] Scaling considerations
  - [ ] Monitoring setup
  - Priority: P2

- [ ] **Developer guide**
  - [ ] Code organization explained
  - [ ] Adding new visualizations
  - [ ] Adding new LLM backends
  - [ ] Testing guidelines
  - Priority: P3

---

## 🚀 Feature Enhancements

### New Visualizations
- [ ] **Word cloud for topics**
  - [ ] Show key concepts
  - [ ] Track concept drift
  - Priority: P3
  - Estimated: 2 days

- [ ] **Timeline view**
  - [ ] Show response evolution over time
  - [ ] Highlight correction points
  - Priority: P3
  - Estimated: 3 days

- [ ] **3D semantic space**
  - [ ] Use PCA with 3 components
  - [ ] Interactive rotation
  - [ ] WebGL rendering
  - Priority: P4
  - Estimated: 1 week

### LLM Backends
- [ ] **Add Anthropic Claude support**
  - [ ] API integration
  - [ ] Handle streaming responses
  - [ ] Test semantic analysis
  - Priority: P3
  - Estimated: 2 days

- [ ] **Add Ollama support (local models)**
  - [ ] API integration
  - [ ] Model selection UI
  - [ ] Performance optimization
  - Priority: P2
  - Estimated: 3 days

- [ ] **Add Azure OpenAI support**
  - [ ] API integration
  - [ ] Configuration handling
  - Priority: P4
  - Estimated: 1 day

### Advanced Features
- [ ] **Multi-model comparison**
  - [ ] Run same question on 2+ models
  - [ ] Compare semantic trajectories
  - [ ] Show model differences
  - Priority: P3
  - Estimated: 2 weeks

- [ ] **Session sharing**
  - [ ] Generate shareable URLs
  - [ ] Public/private toggle
  - [ ] Social media previews
  - Priority: P3
  - Estimated: 1 week

- [ ] **Collaborative sessions**
  - [ ] Real-time updates
  - [ ] Multiple users in same session
  - [ ] WebSocket integration
  - Priority: P4
  - Estimated: 3 weeks

---

## 🐛 Known Issues

### High Priority
- [ ] Safari hard refresh needed to see updates
- [ ] Port 5001 conflicts on some systems
- [ ] Semantic analysis slow with 10+ responses
- [ ] Mobile heatmaps hard to read

### Medium Priority
- [ ] Long responses make UI jump
- [ ] Chart tooltips sometimes off-screen
- [ ] Explainers page back button inconsistent
- [ ] Browser back button doesn't work well

### Low Priority
- [ ] Color accessibility (need colorblind mode)
- [ ] Keyboard navigation incomplete
- [ ] Screen reader support needed
- [ ] Print stylesheet missing

---

## 💡 Ideas / Backlog

### Nice to Have
- [ ] Dark mode
- [ ] Custom color themes
- [ ] Downloadable session history CSV
- [ ] Comparison with previous sessions
- [ ] Keyboard shortcuts
- [ ] Command palette (Cmd+K)
- [ ] Undo/redo for corrections
- [ ] Voice input for questions
- [ ] Auto-save drafts
- [ ] Session templates

### Research / Exploration
- [ ] Can we show attention patterns from API responses?
- [ ] Real-time semantic drift visualization
- [ ] Personality/style analysis of responses
- [ ] Fact-checking integration
- [ ] Citation/source tracking
- [ ] Multi-language support
- [ ] Image input (multimodal)
- [ ] Code execution for programming questions

---

## 📅 Milestone Targets

### November 2025
- [ ] Complete Phase 5 persistence features
- [ ] Fix all P0/P1 bugs
- [ ] Update all documentation
- [ ] Create demo video

### December 2025
- [ ] Launch Phase 6 token prediction demo
- [ ] Achieve 100 test users
- [ ] Publish blog post series

### Q1 2026
- [ ] Complete Phase 6 all demos
- [ ] Mobile experience optimized
- [ ] 1000+ saved sessions

### Q2 2026
- [ ] Launch Phase 7 multi-model comparison
- [ ] Reach 10 educational institutions
- [ ] Active community forming

---

## 🏆 Done Recently

### October 2025
- [x] Implemented comprehensive semantic similarity analysis
- [x] Created beautiful similarity heatmaps
- [x] Added vector movement 2D visualization
- [x] Built sentence-level comparison
- [x] Created interactive explainers (Voting Council, Probability Funnel)
- [x] Added extensive educational content (Theory/Explanation/Calculation boxes)
- [x] Implemented collapsible sections
- [x] Updated to Groq llama-3.1-8b-instant
- [x] Wrote comprehensive README
- [x] Created DEMO_GUIDE

---

## Notes

- **Priority Levels**:
  - P0: Critical (blocks usage)
  - P1: High (important for next release)
  - P2: Medium (nice to have soon)
  - P3: Low (future enhancement)
  - P4: Backlog (might not do)

- **Estimation**:
  - Rough estimates, may change
  - Includes testing time
  - Based on solo development

- **Review Frequency**:
  - Weekly for P0/P1 items
  - Monthly for P2/P3 items
  - Quarterly for P4/backlog

---

*This TODO is a living document. Update as tasks complete and new priorities emerge.*
