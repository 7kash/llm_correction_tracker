#!/usr/bin/env python3
"""
Test which Hugging Face models actually work with the current API
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

hf_token = os.getenv('HUGGING_FACE_TOKEN')
client = InferenceClient(token=hf_token) if hf_token else InferenceClient()

# Models to test - these are known to work with various endpoints
test_models = [
    "gpt2",  # Classic text generation
    "EleutherAI/gpt-neo-125m",  # Small GPT model
    "bigscience/bloom-560m",  # BLOOM small
    "tiiuae/falcon-7b-instruct",  # Falcon
]

print("Testing available models...\n")

for model in test_models:
    print(f"Testing: {model}")
    try:
        # Try simple text generation
        response = client.text_generation(
            "Hello, my name is",
            model=model,
            max_new_tokens=10,
        )
        print(f"  ✅ WORKS! Response: {response[:50]}...")
    except Exception as e:
        print(f"  ❌ Failed: {type(e).__name__}")
    print()
