# 🧪 Local Testing Guide

Step-by-step instructions to test the LLM Inference Tracker on your local machine.

## Prerequisites

- **Python 3.8+** installed
- **4-8GB RAM** available (4GB minimum, 8GB recommended)
- **2-3GB disk space** for model download
- **10-20 minutes** for first-time setup

## Step 1: Check Your Python Version

```bash
python --version
# Should show: Python 3.8.x or higher
# If not, try: python3 --version
```

**If Python < 3.8**, install from [python.org](https://www.python.org/downloads/)

---

## Step 2: Create Virtual Environment (Recommended)

```bash
# Navigate to gradio_app directory
cd gradio_app

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# You should see (venv) in your prompt
```

**Why?** Keeps dependencies isolated from your system Python.

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Collecting gradio==4.19.2...
Collecting transformers==4.37.2...
Collecting torch==2.1.2...
...
Successfully installed gradio-4.19.2 transformers-4.37.2 torch-2.1.2 ...
```

**Time**: 2-5 minutes depending on internet speed

**Common issues:**
- **"No module named pip"**: Run `python -m ensurepip`
- **Slow download**: Try adding `--default-timeout=100` to pip command
- **Permission error**: Use `pip install --user -r requirements.txt`

---

## Step 4: Quick Module Test (Optional but Recommended)

Before running the full app, test individual modules:

```bash
# Test visualization modules
python visualizations/attention_rollout.py
python visualizations/layer_trajectory.py
python visualizations/logit_lens.py
```

**Expected output** for each:
```
✅ Saved [module_name]_test.png
```

**What this does**: Creates test visualizations with synthetic data (no model needed). Verifies matplotlib and numpy work correctly.

**If you see errors**:
- Check that all dependencies installed: `pip list | grep -E "matplotlib|numpy"`
- Try reinstalling: `pip install --upgrade matplotlib numpy`

---

## Step 5: Test LLM Module (First Model Download)

```bash
python backend/llm_with_internals.py
```

**Expected output:**
```
🔧 Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
📍 Using device: cpu  (or cuda/mps if GPU available)
Downloading (…)model.safetensors: 100%|████████| 2.2GB/2.2GB [02:30<00:00, 14.6MB/s]
Downloading (…)tokenizer.json: 100%|████████| 1.8MB/1.8MB [00:01<00:00, 1.2MB/s]
✅ Model loaded: 22 layers

🧪 Testing LLMWithInternals...
📝 Response: Canberra
🔢 Output tokens: 3
👁️ Attention shape: (22, 15, 15)
🧠 Hidden states shape: (22, 2048)
📊 Logits per layer shape: (22, 32000)

🏆 Top-5 tokens at last layer:
  Can: 0.123
  berra: 0.098
  ▁Sydney: 0.067
  ▁capital: 0.045
  ▁Australia: 0.032

✅ Module test passed!
```

**Time**:
- First run: 5-15 minutes (model download)
- Subsequent runs: 10-20 seconds (cached)

**Model cached at**: `~/.cache/huggingface/hub/`

**Common issues:**

### "OutOfMemoryError"
```
RuntimeError: [enforce fail at alloc_cpu.cpp:114] . DefaultCPUAllocator: not enough memory
```
**Solution**: Your system doesn't have enough RAM. Options:
1. Close other apps to free memory
2. Use a smaller model (GPT-2): Edit `backend/llm_with_internals.py` line 31:
   ```python
   model_name: str = "gpt2"  # Only 500MB!
   ```
3. Restart your computer and try again

### "Connection timeout"
```
requests.exceptions.ConnectTimeout: HTTPSConnectionPool...
```
**Solution**: Slow internet. Increase timeout:
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=600
python backend/llm_with_internals.py
```

### "CUDA out of memory" (GPU users)
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solution**: Force CPU mode:
```python
# In backend/llm_with_internals.py, line 36, change:
device: str = "cpu"  # Force CPU
```

---

## Step 6: Run the Full Gradio App

```bash
python app.py
```

**Expected output:**
```
🔧 Loading TinyLlama model...
📍 Using device: cpu
✅ Model ready!

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live (temporary)

To create a permanent link, set `share=True` in `launch()`.
```

**Browser opens automatically** (if not, click the URL)

---

## Step 7: Interactive Testing Checklist

### Test 1: Basic Response Generation

1. In the question box, type:
   ```
   What is the capital of Australia?
   ```

2. Click **"🚀 Generate Response"**

3. **Expected**:
   - Wait 2-5 seconds (first generation may be slower)
   - Response appears: Something like "The capital of Australia is Canberra" or "Sydney" (depends on model)
   - Turn summary shows: "Turn 0" with token count

4. **If error**: Check terminal for error messages

---

### Test 2: Attention Rollout Visualization

1. Click **"🎯 Attention Rollout"** tab

2. Set **"Select Turn to Visualize"** slider to **0**

3. Click **"🔍 Show Attention"**

4. **Expected**:
   - Visualization appears in ~1-2 seconds
   - Shows ribbon diagram with tokens on left
   - Thicker ribbons = more attention
   - Colors from cool (blue) to warm (red/yellow)

5. **Interpretation**:
   - See which input words the model focused on
   - Top contributors listed at bottom of plot

---

### Test 3: Layer Trajectory Visualization

1. Click **"📈 Layer Trajectories"** tab

2. Set slider to **0**, click **"🔍 Show Trajectory"**

3. **Expected**:
   - 2D plot showing line from L0 to L11 (or L21 for TinyLlama)
   - Points represent layers
   - Shows how hidden state evolved

4. **Interpretation**:
   - Line = token's journey through model layers
   - Longer line = more transformation
   - Annotated with "L0" (start) and final layer

---

### Test 4: Logit Lens Visualization

1. Click **"🔍 Logit Lens"** tab

2. Set slider to **0**, select **"heatmap"** mode

3. Click **"🔍 Show Logit Lens"**

4. **Expected**:
   - Heatmap showing top-5 tokens per layer
   - Each cell has token name + probability
   - Warmer colors = higher probability

5. **Interpretation**:
   - See what model "wanted to say" at each layer
   - Watch prediction form from layer 0 → final
   - Top token may change across layers

---

### Test 5: Correction & Comparison

1. Go back to main question box

2. Type correction:
   ```
   Actually, the capital is Canberra, not Sydney.
   ```

3. Click **"🚀 Generate Response"**

4. **Expected**:
   - New response generated (Turn 1)
   - Turn summary shows **"Changes from previous turn"**
   - Lists attention shift and prediction changes

5. **Compare visualizations**:
   - View Turn 0 attention → Turn 1 attention
   - Notice which tokens gained/lost attention
   - In trajectories tab: Set slider to 1, see divergence

---

## Step 8: Test Edge Cases

### Empty Question
- Type nothing, click generate
- **Expected**: "Please enter a question!" message

### Very Long Question
- Type 200+ words
- **Expected**: Works, but may be slower (10-15s)

### Reset Session
- Click **"🔄 Reset Session"**
- **Expected**: "Session reset!" message, history cleared

### Multiple Corrections
- Ask question → correct 3-4 times
- **Expected**: All turns tracked, can visualize any turn

---

## Step 9: Performance Check

**Good performance**:
- First generation: 2-5 seconds
- Subsequent generations: 2-3 seconds
- Visualizations: <1 second each

**Slow performance** (>10s per generation):
- **Cause**: CPU inference is slower
- **Solutions**:
  1. Use smaller model (GPT-2)
  2. Reduce `max_new_tokens` in `app.py` line 38: `max_tokens: int = 50`
  3. Use GPU if available

**Check if GPU is being used**:
```python
# In terminal where app is running, you should see:
# 📍 Using device: cuda  (or mps for Mac M1/M2)
# If it says 'cpu', your GPU isn't detected
```

---

## Step 10: Stop the App

```bash
# In terminal where app is running:
Ctrl + C

# Deactivate virtual environment:
deactivate
```

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'gradio'"
**Solution**: Activate venv, reinstall requirements
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Port 7860 already in use"
**Solution**: Kill existing process
```bash
# macOS/Linux:
lsof -ti:7860 | xargs kill -9

# Windows:
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Or change port in app.py:
# demo.launch(server_port=7861)
```

### Issue: Visualizations show blank/empty
**Solution**:
1. Check terminal for errors
2. Try increasing turn slider range (may be set too low)
3. Ensure you generated at least 1 response first

### Issue: App loads but no response generated
**Solution**:
1. Check RAM usage (Task Manager/Activity Monitor)
2. Try GPT-2 instead of TinyLlama
3. Check terminal for error stack trace

### Issue: Model download stuck at 0%
**Solution**:
```bash
# Clear cache and retry:
rm -rf ~/.cache/huggingface/
python backend/llm_with_internals.py
```

---

## ✅ Success Checklist

After testing, you should have:

- [x] Model downloaded and cached (~2GB)
- [x] Generated at least 2 responses (question + correction)
- [x] Viewed attention rollout for both turns
- [x] Viewed layer trajectories for both turns
- [x] Viewed logit lens heatmap
- [x] Saw "Changes from previous turn" summary
- [x] Compared before/after visualizations

**If all checked** ✅ You're ready to deploy to HuggingFace Spaces!

---

## 📊 Example Test Session (Copy-Paste)

```
1. Question: What is the capital of Australia?
   Response: Sydney / Canberra (depends on model)

2. Correction: Actually, it's Canberra, not Sydney.
   Response: You're right, Canberra is the capital of Australia.

3. View Turn 0 attention → See focus on "Australia", "capital"
4. View Turn 1 attention → See focus shift to "Canberra", "correction"
5. Compare trajectories → Notice divergence at layer 8-10
6. Logit lens → See prediction change from "Sydney" to "Canberra"
```

---

## 🎉 Next Steps After Local Testing

1. **Works great?** → Deploy to HuggingFace Spaces
2. **Too slow?** → Try GPT-2 or optimize model
3. **Found bugs?** → Check `app.py` and visualization modules
4. **Want more features?** → See `CLAUDE.md` for development guide

Need help with any step? Check the terminal output for detailed error messages!
