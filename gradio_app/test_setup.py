#!/usr/bin/env python3
"""
Quick setup verification script

Run this before launching the full app to verify:
1. All dependencies installed
2. Model downloads successfully
3. Visualizations work
4. Basic inference works
"""

import sys
import subprocess
from pathlib import Path

print("=" * 60)
print("🧪 LLM Inference Tracker - Setup Verification")
print("=" * 60)
print()

# Step 1: Check Python version
print("Step 1/6: Checking Python version...")
version = sys.version_info
if version.major < 3 or (version.major == 3 and version.minor < 8):
    print(f"❌ Python {version.major}.{version.minor} detected")
    print("   Requires Python 3.8 or higher")
    print("   Download from: https://www.python.org/downloads/")
    sys.exit(1)
else:
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
print()

# Step 2: Check dependencies
print("Step 2/6: Checking dependencies...")
required = ["gradio", "transformers", "torch", "numpy", "matplotlib"]
missing = []

for pkg in required:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg}")
        missing.append(pkg)

if missing:
    print()
    print(f"❌ Missing packages: {', '.join(missing)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

print()

# Step 3: Test visualization modules
print("Step 3/6: Testing visualization modules...")

try:
    from visualizations import attention_rollout, layer_trajectory, logit_lens
    print("  ✅ All visualization modules import successfully")
except Exception as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)

print()

# Step 4: Test synthetic visualizations
print("Step 4/6: Testing synthetic data visualizations...")
print("  (This creates test plots with fake data, no model needed)")
print()

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Test attention rollout
    print("  Testing attention rollout...")
    np.random.seed(42)
    num_layers, seq_len = 8, 10
    tokens = [f"tok{i}" for i in range(seq_len)]
    attn = np.random.rand(num_layers, seq_len, seq_len)
    attn = attn / attn.sum(axis=2, keepdims=True)

    fig = attention_rollout.plot_attention_rollout(
        attn, tokens, target_pos=-1, top_k=5
    )
    plt.close(fig)
    print("    ✅ Attention rollout works")

    # Test layer trajectory
    print("  Testing layer trajectory...")
    hidden = np.random.randn(num_layers, 64)
    fig = layer_trajectory.plot_layer_trajectory(hidden, "test_token")
    plt.close(fig)
    print("    ✅ Layer trajectory works")

    # Test logit lens
    print("  Testing logit lens...")
    logits = np.random.randn(num_layers, 100)
    vocab = [f"token_{i}" for i in range(100)]
    tokens_per_layer, probs = logit_lens.get_top_k_per_layer(logits, vocab, k=5)
    fig = logit_lens.plot_logit_lens_heatmap(tokens_per_layer, probs)
    plt.close(fig)
    print("    ✅ Logit lens works")

except Exception as e:
    print(f"  ❌ Visualization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 5: Test model loading (this is the long one)
print("Step 5/6: Testing model loading...")
print("  ⏳ This will download TinyLlama (~2GB) on first run...")
print("  ⏳ Estimated time: 2-10 minutes depending on internet speed")
print("  ⏳ Subsequent runs will be much faster (model cached)")
print()

try:
    from backend.llm_with_internals import LLMWithInternals

    print("  Loading model...")
    llm = LLMWithInternals(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("  ✅ Model loaded successfully")
    print(f"     Device: {llm.device}")
    print(f"     Layers: {llm.model.config.num_hidden_layers}")

except Exception as e:
    print(f"  ❌ Model loading failed: {e}")
    print()
    print("Common issues:")
    print("  - Not enough RAM: Need 4GB+ free")
    print("  - Internet timeout: Try again with better connection")
    print("  - Disk space: Need 2-3GB free")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 6: Test inference
print("Step 6/6: Testing inference with real model...")
print("  ⏳ Generating response (this may take 5-10 seconds)...")
print()

try:
    result = llm.generate_with_internals(
        question="What is 2+2?",
        max_new_tokens=10,
        temperature=0.7
    )

    print(f"  ✅ Response: {result['response'][:50]}...")
    print(f"  ✅ Attention shape: {result['attentions'].shape}")
    print(f"  ✅ Hidden states shape: {result['hidden_states'].shape}")
    print(f"  ✅ Logits per layer shape: {result['logits_per_layer'].shape}")

    # Test top-k extraction
    logits = result['logits_per_layer'][-1]
    tokens, probs = llm.get_top_k_tokens(logits, k=3)
    print(f"  ✅ Top-3 tokens: {tokens}")

except Exception as e:
    print(f"  ❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("🎉 All tests passed!")
print("=" * 60)
print()
print("✅ Setup verification complete")
print()
print("Next steps:")
print("  1. Run the app: python app.py")
print("  2. Open browser: http://localhost:7860")
print("  3. Try asking: 'What is the capital of Australia?'")
print("  4. Then correct: 'Actually it's Canberra'")
print("  5. Explore visualizations!")
print()
print("See TESTING_GUIDE.md for detailed testing instructions.")
print()
