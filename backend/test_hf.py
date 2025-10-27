#!/usr/bin/env python3
"""
Test script to verify Hugging Face API connection
Run this to diagnose issues before using in the main app
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("HUGGING FACE CONNECTION TEST")
print("=" * 60)

# Check environment variables
hf_token = os.getenv('HUGGING_FACE_TOKEN')
hf_model = os.getenv('HUGGING_FACE_MODEL', 'google/flan-t5-base')

print(f"\n1. Environment Check:")
print(f"   Token present: {'Yes' if hf_token else 'No'}")
if hf_token:
    print(f"   Token preview: {hf_token[:10]}...{hf_token[-5:]}")
print(f"   Model: {hf_model}")

# Try importing huggingface_hub
print(f"\n2. Library Check:")
try:
    from huggingface_hub import InferenceClient
    print(f"   ✅ huggingface_hub imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import: {e}")
    exit(1)

# Create client
print(f"\n3. Client Creation:")
try:
    if hf_token:
        client = InferenceClient(token=hf_token)
        print(f"   ✅ Client created with token")
    else:
        client = InferenceClient()
        print(f"   ✅ Client created without token (public API)")
except Exception as e:
    print(f"   ❌ Failed to create client: {e}")
    exit(1)

# Test simple text generation with FLAN-T5
print(f"\n4. Testing FLAN-T5 (most reliable model):")
test_model = "google/flan-t5-base"
try:
    response = client.text_generation(
        "Answer this question: What is 2+2?",
        model=test_model,
        max_new_tokens=50,
    )
    print(f"   ✅ SUCCESS!")
    print(f"   Question: What is 2+2?")
    print(f"   Answer: {response}")
except Exception as e:
    print(f"   ❌ Failed: {type(e).__name__}: {e}")
    import traceback
    print(f"\n   Full traceback:")
    traceback.print_exc()

# Test with your configured model
if hf_model != test_model:
    print(f"\n5. Testing your configured model ({hf_model}):")
    try:
        response = client.text_generation(
            "Answer this question: What is 2+2?",
            model=hf_model,
            max_new_tokens=50,
        )
        print(f"   ✅ SUCCESS!")
        print(f"   Answer: {response}")
    except Exception as e:
        print(f"   ❌ Failed: {type(e).__name__}: {e}")
        print(f"   Recommendation: Stick with google/flan-t5-base")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
