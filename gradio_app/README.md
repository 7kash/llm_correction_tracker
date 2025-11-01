# 🧠 LLM Inference Tracker (Gradio App)

**Track how Large Language Models change their responses when corrected** — with visualizations of internal mechanisms.

## 🎯 What It Does

1. **Ask a one-word question** → LLM generates response
2. **Provide correction feedback** → LLM adjusts response
3. **Visualize internals**:
   - 🎯 **Attention Distribution**: Which words the model focused on
   - 🎲 **Softmax Probabilities**: Top token candidates and their probabilities
   - 📊 **Layer-by-Layer Predictions**: How the answer forms through all 22 layers

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

### 🎯 Attention Distribution
Shows which input words the model focused on when generating the answer.

**How to read**:
- Horizontal bars showing attention percentage for each word
- Higher percentage = more focus on that word
- Real percentages from model's final layer (sum to 100%)

### 🎲 Top Token Probabilities
Shows the softmax probability distribution for the final token prediction.

**How to read**:
- Top 5 most likely tokens with their logits and probabilities
- Real softmax probabilities from the model
- Shows what alternatives the model considered

### 📊 Layer-by-Layer Predictions
Uses the "logit lens" technique to show what the model predicts at each layer.

**How to read**:
- All 22 layers shown from early (uncertain) to final (confident)
- Each layer shows actual answer probability + top 3 alternatives
- Watch how confidence increases through layers

## 🎓 Educational Use Cases

### For Teachers
- Demonstrate attention mechanisms with real examples
- Show how corrections change model behavior
- Explain layer-by-layer refinement

### For Students
- Hands-on exploration of LLM internals
- Visual understanding of "how the model thinks"
- Compare before/after corrections side-by-side

### For Researchers
- Quick prototyping tool for analyzing model behavior
- Extract attention/predictions for experiments
- Logit lens analysis out-of-the-box

## 🔧 Technical Details

### Model
- **TinyLlama-1.1B-Chat-v1.0** (default)
- 22 layers, 2048 hidden dim, 32 attention heads
- Chat-tuned for conversational use

### Internals Extracted
- **Attention weights**: (layers, heads, seq, seq) → mean-over-heads → percentages
- **Layer predictions**: Logits from each layer passed through LM head
- **Softmax probabilities**: Real probability distribution over vocabulary

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
│   └── answer_flow.py              # Token cleaning utilities
└── README.md                       # This file
```

## 🤔 Common Questions

**Q: Why TinyLlama?**
A: Good balance of quality and speed. Runs on free HF Spaces CPU tier.

**Q: Can I use other models?**
A: Yes! Edit `model_name` in `app.py`. Any HuggingFace CausalLM works.

**Q: How accurate are visualizations?**
A: 100% accurate - extracted directly from model internals, not approximations.

**Q: What questions work best?**
A: One-word answerable questions like "What is the capital of France?" or "What color is the sky?"

## 🚀 Deployment Options

| Platform | Cost | Speed | Notes |
|----------|------|-------|-------|
| **HF Spaces (CPU)** | FREE | ~3-5s | May queue when busy |
| **HF Spaces (GPU)** | $70/mo | ~1-2s | Always-on, no queue |
| **Local** | $0 | Depends on HW | Best for development |

**Recommendation**: Start with HF Spaces CPU (free), upgrade to GPU if popular.

## 📚 Learn More

- **Logit Lens**: [nostalgebraist blog](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- **Attention Mechanisms**: [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- **TinyLlama**: [Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## 🤝 Contributing

Improvements welcome! Areas:
- Support for more models (Llama, Mistral, etc.)
- Additional visualizations (token-by-token generation)
- Session persistence (SQLite)
- Export visualizations as images

## 📄 License

MIT License - Free for educational and research use!

---

**Built with**: Gradio • Transformers • PyTorch • NumPy

**Making LLM internals accessible through interactive visualizations** ✨
