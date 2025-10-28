# ⚡ Quick Start - Local Testing

**Time**: 15-20 minutes (first run)
**Prerequisites**: Python 3.8+, 4GB RAM, 2GB disk space

---

## 🚀 Fast Track (4 Commands)

```bash
# 1. Navigate to app
cd gradio_app

# 2. Create virtual environment (recommended)
python -m venv venv && source venv/bin/activate  # Linux/Mac
# python -m venv venv && venv\Scripts\activate    # Windows

# 3. Install dependencies (2-5 min)
pip install -r requirements.txt

# 4. Run verification test (5-10 min first time, downloads model)
python test_setup.py
```

**If all tests pass ✅**, run the app:

```bash
python app.py
# Open http://localhost:7860
```

---

## 📝 Test the App (5 minutes)

### 1. Ask a Question
```
What is the capital of Australia?
```
Click **"🚀 Generate Response"** → Wait 2-5s

### 2. View Attention
Go to **"🎯 Attention Rollout"** tab → Click **"🔍 Show Attention"**
- See which words the model focused on

### 3. View Trajectories
Go to **"📈 Layer Trajectories"** tab → Click **"🔍 Show Trajectory"**
- See how representation evolved through layers

### 4. View Logit Lens
Go to **"🔍 Logit Lens"** tab → Select **"heatmap"** → Click **"🔍 Show Logit Lens"**
- See what model "wanted to say" at each layer

### 5. Provide Correction
```
Actually, the capital is Canberra, not Sydney.
```
Click **"🚀 Generate Response"** → Compare visualizations!

---

## 🐛 Troubleshooting (30 seconds)

| Problem | Solution |
|---------|----------|
| "No module named gradio" | `pip install -r requirements.txt` |
| "Port 7860 in use" | Change port in app.py or kill process |
| Model download slow/stuck | Wait or clear cache: `rm -rf ~/.cache/huggingface/` |
| "Out of memory" | Close apps or use GPT-2 (edit `backend/llm_with_internals.py` line 31) |
| Slow inference (>10s) | Normal for CPU, use GPU or smaller model |

---

## ✅ Success Checklist

- [x] `test_setup.py` passes all 6 steps
- [x] App opens in browser at localhost:7860
- [x] Generated 2+ responses (question + correction)
- [x] Viewed all 3 visualization types
- [x] Turn summary shows "Changes from previous turn"

**All checked?** 🎉 Ready to deploy to HuggingFace Spaces!

---

## 📚 More Details

- **Full testing guide**: See `TESTING_GUIDE.md`
- **Deployment**: See `README.md`
- **Code structure**: See `../CLAUDE.md`

---

## 🆘 Need Help?

1. Run `python test_setup.py` and share output
2. Check terminal where `app.py` is running for errors
3. Try the example question first: "What is the capital of Australia?"

**Common first-time issues**:
- Model download takes 5-10 min (only once!)
- First inference is slower (~5-10s)
- Browser cache: Hard refresh (Cmd+Shift+R / Ctrl+Shift+R)
