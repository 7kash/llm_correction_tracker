#!/usr/bin/env python3
"""
Test script for layer-by-layer visualization (logit lens).

Run this to see how the model's prediction forms through layers.
"""

from backend.llm_with_internals import LLMWithInternals
from visualizations.answer_flow import create_layer_by_layer_visualization

def main():
    print("🔧 Loading TinyLlama model...")
    model = LLMWithInternals(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("✅ Model loaded!\n")

    # Test questions
    questions = [
        "What is the capital of Australia?",
        "What is the capital of France?",
        "What is 2+2?",
    ]

    for question in questions:
        print("="*70)
        print(f"QUESTION: {question}")
        print("="*70)

        # Generate with layer predictions
        result = model.generate_one_word_with_layers(question, max_new_tokens=3)

        # Get vocab for visualization
        vocab_dict = model.tokenizer.get_vocab()
        vocab_size = len(vocab_dict)
        vocab = [""] * vocab_size
        for token_str, token_id in vocab_dict.items():
            vocab[token_id] = token_str

        # Create visualization
        visualization = create_layer_by_layer_visualization(result, vocab)

        print(visualization)
        print("\n")


if __name__ == "__main__":
    main()
