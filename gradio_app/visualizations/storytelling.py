"""
Enhanced visualization helpers for Phase 1 improvements.

Adds storytelling and insights to make visualizations more educational.
"""

import numpy as np
from typing import List, Tuple, Dict


def get_token_by_token_attention_story(
    attentions: np.ndarray,
    input_tokens: List[str],
    output_tokens: List[str],
    top_k: int = 3
) -> str:
    """
    Create a narrative of what each output token attended to.

    Parameters
    ----------
    attentions : np.ndarray
        Shape (num_layers, seq_len, seq_len) - mean over heads
    input_tokens : list[str]
        Input/prompt tokens
    output_tokens : list[str]
        Generated output tokens
    top_k : int
        Number of top attended tokens to show per output token

    Returns
    -------
    story : str
        Markdown-formatted narrative
    """
    num_layers = attentions.shape[0]
    all_tokens = input_tokens + output_tokens
    input_len = len(input_tokens)

    story_parts = ["## 📖 Token-by-Token Attention Story\n"]
    story_parts.append("_What each generated token focused on:_\n")

    # Use attention from last layer (most refined)
    last_layer_attn = attentions[-1]  # (seq, seq)

    for i, out_token in enumerate(output_tokens[:10]):  # Limit to first 10 tokens
        token_pos = input_len + i

        # Get attention weights for this output token
        attn_weights = last_layer_attn[token_pos, :]  # Attention to all previous tokens

        # Get top-k attended tokens
        top_indices = np.argsort(attn_weights)[-top_k:][::-1]

        story_parts.append(f"### Token #{i+1}: `{out_token}`")

        for rank, idx in enumerate(top_indices, 1):
            if idx < len(all_tokens):
                attended_token = all_tokens[idx]
                weight = attn_weights[idx]
                percentage = weight * 100

                # Add interpretation
                emoji = "🎯" if rank == 1 else "📍"
                story_parts.append(f"- {emoji} **{percentage:.1f}%** attention on `{attended_token}`")

        # Add interpretation based on attention pattern
        if top_indices[0] < input_len:
            story_parts.append(f"  - 💡 _Referencing input: '{all_tokens[top_indices[0]]}'_")
        else:
            story_parts.append(f"  - 💡 _Building on previous output_")

        story_parts.append("")  # Blank line

    if len(output_tokens) > 10:
        story_parts.append(f"_... and {len(output_tokens) - 10} more tokens_\n")

    return "\n".join(story_parts)


def analyze_logit_lens_story(
    logits_per_layer: np.ndarray,
    vocab: List[str],
    top_k: int = 3
) -> Dict:
    """
    Analyze logit lens to find the "aha moment" where prediction changes.

    Parameters
    ----------
    logits_per_layer : np.ndarray
        Shape (num_layers, vocab_size)
    vocab : list[str]
        Vocabulary indexed by token ID
    top_k : int
        Number of top predictions to track

    Returns
    -------
    analysis : dict
        Contains:
        - phases: List of (layer_range, top_token, probability, description)
        - aha_moment_layer: int or None
        - story: str (markdown narrative)
    """
    num_layers = logits_per_layer.shape[0]

    # Compute probabilities per layer
    probs_per_layer = []
    top_tokens_per_layer = []

    for logits in logits_per_layer:
        # Softmax
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        # Get top token
        top_idx = np.argmax(probs)
        top_token = vocab[top_idx] if top_idx < len(vocab) else f"<{top_idx}>"
        top_prob = probs[top_idx]

        probs_per_layer.append(top_prob)
        top_tokens_per_layer.append(top_token)

    # Find where top token changes (aha moments)
    aha_moments = []
    for l in range(1, num_layers):
        if top_tokens_per_layer[l] != top_tokens_per_layer[l-1]:
            aha_moments.append(l)

    # Group into phases
    phases = []
    phase_start = 0

    for aha_layer in aha_moments + [num_layers]:
        if aha_layer > phase_start:
            phase_token = top_tokens_per_layer[phase_start]
            phase_prob = np.mean(probs_per_layer[phase_start:aha_layer])

            phases.append({
                "layer_range": (phase_start, aha_layer - 1),
                "top_token": phase_token,
                "probability": phase_prob,
                "is_final": (aha_layer == num_layers)
            })

            phase_start = aha_layer

    # Create story
    story_parts = ["## 💡 Logit Lens: Prediction Journey\n"]
    story_parts.append("_How the model's prediction evolved layer by layer:_\n")

    for i, phase in enumerate(phases):
        start, end = phase["layer_range"]
        token = phase["top_token"]
        prob = phase["probability"]

        if i == 0:
            emoji = "🌱"
            desc = "**Initial thought**"
        elif phase["is_final"]:
            emoji = "✅"
            desc = "**Final answer**"
        else:
            emoji = "🔄"
            desc = "**Reconsidering**"

        story_parts.append(f"### {emoji} Layers {start}-{end}: {desc}")
        story_parts.append(f"- Top prediction: `{token}` ({prob*100:.1f}% confident)")

        # Add interpretation
        if i == 0:
            story_parts.append(f"  - 💭 _Model's gut feeling_")
        elif not phase["is_final"]:
            story_parts.append(f"  - ⚡ **AHA MOMENT at Layer {start}!** Prediction changed!")
        else:
            story_parts.append(f"  - 🎯 _High confidence - answer locked in_")

        story_parts.append("")

    # Highlight the key aha moment
    main_aha = aha_moments[0] if aha_moments else None

    if main_aha:
        story_parts.append(f"### 🎯 Key Insight")
        story_parts.append(f"The **critical shift** happened at **Layer {main_aha}**")
        story_parts.append(f"- Before: `{top_tokens_per_layer[main_aha-1]}`")
        story_parts.append(f"- After: `{top_tokens_per_layer[main_aha]}`")
        story_parts.append(f"- This is where the model \"figured it out\"!")

    return {
        "phases": phases,
        "aha_moment_layer": main_aha,
        "story": "\n".join(story_parts),
        "top_tokens_per_layer": top_tokens_per_layer,
        "probs_per_layer": probs_per_layer
    }


def create_enhanced_turn_summary(
    internals: dict,
    previous_internals: dict = None
) -> str:
    """
    Create enhanced turn summary with actionable insights.

    Parameters
    ----------
    internals : dict
        Current turn internals
    previous_internals : dict
        Previous turn internals (for comparison)

    Returns
    -------
    summary : str
        Markdown-formatted enhanced summary
    """
    summary_parts = []

    # Basic info
    num_tokens = len(internals["tokens"])
    num_layers = internals["num_layers"]

    summary_parts.append(f"# 📊 Turn Summary\n")
    summary_parts.append(f"**Generated**: {num_tokens} tokens across {num_layers} layers\n")

    if previous_internals:
        summary_parts.append("## 🔄 What Changed?\n")

        # Token attention comparison
        curr_tokens = internals["input_tokens"] + internals["tokens"]
        prev_tokens = previous_internals["input_tokens"] + previous_internals["tokens"]

        # Find new tokens that gained attention
        new_tokens = set(curr_tokens) - set(prev_tokens)
        if new_tokens:
            summary_parts.append("### 🆕 New Focus Areas")
            for tok in list(new_tokens)[:5]:
                summary_parts.append(f"- Model now pays attention to: `{tok}`")
            summary_parts.append("")

        summary_parts.append("### 💡 Key Insight")
        summary_parts.append("When you provide a **correction**, the model:")
        summary_parts.append("1. Detects correction signal words ('Actually', 'correction')")
        summary_parts.append("2. Shifts attention from general knowledge → your specific input")
        summary_parts.append("3. Changes its prediction at a specific layer (the 'aha moment')")
        summary_parts.append("4. Builds confidence in the corrected answer\n")

    else:
        summary_parts.append("_This is the first turn. Generate a correction to see how the model changes!_\n")

    return "\n".join(summary_parts)


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing visualization helpers...")

    # Test token-by-token attention
    num_layers = 12
    seq_len = 15
    attentions = np.random.rand(num_layers, seq_len, seq_len)
    # Normalize
    attentions = attentions / attentions.sum(axis=2, keepdims=True)

    input_tokens = ["What", "is", "the", "capital", "of", "Australia", "?"]
    output_tokens = ["Can", "berra", "is", "the", "capital"]

    story = get_token_by_token_attention_story(attentions, input_tokens, output_tokens)
    print(story)
    print("\n" + "="*60 + "\n")

    # Test logit lens analysis
    logits = np.random.randn(num_layers, 100)
    vocab = [f"token_{i}" for i in range(100)]
    vocab[10] = "Sydney"
    vocab[25] = "Canberra"

    # Make it change from Sydney to Canberra at layer 8
    logits[:8, 10] += 3.0  # Sydney boost in early layers
    logits[8:, 25] += 4.0  # Canberra boost in later layers

    analysis = analyze_logit_lens_story(logits, vocab)
    print(analysis["story"])

    print("\n✅ Tests complete!")
