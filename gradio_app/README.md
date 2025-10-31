# 🧠 LLM Inference Tracker (Gradio App)

**Track how Large Language Models change their responses when corrected** — with visualizations of internal mechanisms.

## 🎯 What It Does

1. **Ask a question** → LLM generates response
2. **Provide correction** → LLM adjusts response
3. **Visualize internals**:
   - 🎯 **Attention Rollout**: Which words the model focused on
   - 📈 **Layer Trajectories**: How representations evolved through layers
   - 🔍 **Logit Lens**: What the model "wanted to say" at each layer

## 🚀 Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
python app.py

# Open browser to http://localhost:7860
```

**First run**: Downloads TinyLlama model (~2GB, one-time)

### Deploy to Hugging Face Spaces

1. Create new Space at https://huggingface.co/new-space
2. Select: **Gradio** SDK
3. Upload files:
   - `app.py`
   - `requirements.txt`
   - `backend/` folder
   - `visualizations/` folder
4. Space builds automatically!

**Hardware**: CPU Basic (free) works, but GPU is faster

## 📊 Visualizations Explained

### 🎯 Attention Rollout
Shows which input tokens contributed most to the final prediction by propagating attention backward through layers.

**How to read**:
- Ribbons = attention flow
- Thicker ribbon = more contribution
- Colors: cool (low) → warm (high)

### 📈 Layer Trajectories
Tracks how token representations move through 2D PCA space across layers.

**How to read**:
- Lines = token's journey through layers
- Divergence between before/after = where correction took effect
- L0 = input, L11 = final layer

### 🔍 Logit Lens
Shows what the model "wants to say" at each intermediate layer by applying the final prediction head.

**How to read**:
- Heatmap: Top-k tokens per layer
- Evolution plot: Track specific tokens across layers
- See when prediction changes from wrong → correct

## 🎓 Educational Use Cases

### For Teachers
- Demonstrate attention mechanisms with real examples
- Show how corrections propagate through layers
- Explain logit lens concept interactively

### For Students
- Hands-on exploration of LLM internals
- Visual understanding of "how the model thinks"
- Compare before/after corrections

### For Researchers
- Quick prototyping tool for analyzing model behavior
- Extract attention/hidden states for experiments
- Logit lens analysis out-of-the-box

## 🔧 Technical Details

### Model
- **TinyLlama-1.1B-Chat-v1.0** (default)
- 22 layers, 2048 hidden dim, 32 attention heads
- Chat-tuned for conversational use

### Internals Extracted
- **Attention weights**: (layers, heads, seq, seq) → mean-over-heads
- **Hidden states**: (layers, hidden_dim) for final token
- **Logits per layer**: (layers, vocab_size) via LM head

### Compute Requirements
- **RAM**: ~4GB minimum (model + activations)
- **GPU**: Optional (2-3x faster)
- **Inference time**: 2-5 seconds per response (CPU)

## 📂 Project Structure

```
gradio_app/
├── app.py                          # Main Gradio interface
├── requirements.txt                # Dependencies
├── backend/
│   └── llm_with_internals.py      # Model wrapper with extraction
├── visualizations/
│   ├── attention_rollout.py        # Attention flow diagrams
│   ├── layer_trajectory.py         # Hidden state evolution
│   └── logit_lens.py               # Per-layer predictions
└── README.md                       # This file
```

## 🤔 Common Questions

**Q: Why TinyLlama?**
A: Good balance of quality and speed. Runs on free HF Spaces CPU tier.

**Q: Can I use other models?**
A: Yes! Edit `model_name` in `app.py`. Any HuggingFace CausalLM works.

**Q: How accurate are visualizations?**
A: 100% accurate - extracted directly from model internals, not approximations.

**Q: Can I save examples?**
A: Currently session-only (resets on reload). Persistence coming in future version.

## 🚀 Deployment Options

| Platform | Cost | Speed | Notes |
|----------|------|-------|-------|
| **HF Spaces (CPU)** | FREE | ~3-5s | May queue when busy |
| **HF Spaces (GPU)** | $70/mo | ~1-2s | Always-on, no queue |
| **Local** | $0 | Depends on HW | Best for development |

**Recommendation**: Start with HF Spaces CPU (free), upgrade to GPU if popular.

## 📚 Learn More

- **Attention Rollout**: [Abnar & Zuidema (2020)](https://arxiv.org/abs/2005.00928)
- **Logit Lens**: [nostalgebraist blog](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- **TinyLlama**: [Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## 🤝 Contributing

Improvements welcome! Areas:
- Support for more models (Llama, Mistral, etc.)
- Additional visualizations (beam search tree, token probabilities)
- Session persistence (SQLite)
- Export visualizations as PNG/PDF

## 📄 License

MIT License - Free for educational and research use!

---

**Built with**: Gradio • Transformers • PyTorch • Matplotlib • NumPy

**Making LLM internals accessible through interactive visualizations** ✨
