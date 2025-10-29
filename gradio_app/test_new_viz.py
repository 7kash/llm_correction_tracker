#!/usr/bin/env python3
"""
Test the NEW layer-by-layer visualization with actual answer tracking.
Run this to verify the fix is working.
"""

import sys
sys.dont_write_bytecode = True  # Prevent .pyc files

from backend.llm_with_internals import LLMWithInternals
from visualizations.answer_flow import create_layer_by_layer_visualization

print("="*70)
print("TESTING NEW VISUALIZATION - ACTUAL ANSWER TRACKING")
print("="*70)
print("\n🔧 Loading TinyLlama model...\n")

model = LLMWithInternals(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("✅ Model loaded!")
print("\n" + "="*70)
print("QUESTION: What is 2+2?")
print("="*70 + "\n")

# Generate with layer predictions
result = model.generate_one_word_with_layers("What is 2+2?", max_new_tokens=3)

print(f"Generated answer: {result['response']}")
print(f"Number of layers: {result['num_layers']}\n")

# Check the format of predictions
first_layer = result['layer_predictions'][0]
print(f"First layer predictions format: {first_layer['predictions'][0].keys()}")
print(f"Has 'is_actual_answer' flag: {'is_actual_answer' in first_layer['predictions'][0]}")
print()

# Get vocab
vocab_dict = model.tokenizer.get_vocab()
vocab_size = len(vocab_dict)
vocab = [""] * vocab_size
for token_str, token_id in vocab_dict.items():
    vocab[token_id] = token_str

# Create visualization
visualization = create_layer_by_layer_visualization(result, vocab)

print(visualization)
print("\n" + "="*70)
print("✅ Test complete!")
print("="*70)
