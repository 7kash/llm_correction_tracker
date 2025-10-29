"""
Answer Generation Flow Visualization

Shows HOW the LLM generated its answer in 3 clear stages:
1. Input Understanding
2. Knowledge Retrieval
3. Answer Formation
"""

import numpy as np
from typing import List, Dict, Tuple
import re


def clean_token(token: str) -> tuple:
    """
    Clean tokenizer artifacts from tokens.

    Returns
    -------
    (cleaned_token, has_space) : tuple or None
        cleaned_token: str without artifacts
        has_space: bool indicating if this starts a new word
        Returns None if token is just noise
    """
    # Remove these entirely - be very aggressive with chat template tokens
    if not token or len(token) == 0:
        return None

    # Remove any chat template markers from the token first
    # This handles cases like "Australia<|assistant|>" as one token
    chat_markers = ['<|user|>', '<|assistant|>', '<|im_start|>', '<|im_end|>', '<|system|>']
    for marker in chat_markers:
        token = token.replace(marker, '')

    # After removing markers, check if anything is left
    if not token:
        return None

    # Direct string matches for chat template (in case token IS just the marker)
    if token in chat_markers:
        return None

    # Pattern-based filtering - match anywhere in string, not just start
    noise_patterns = [
        r'<[^>]*>',         # Anything in angle brackets
        r'^▁+$',            # Just underscores
        r'^\s+$',           # Just whitespace
        r'^[<>|]+$',        # Just brackets and pipes
    ]

    for pattern in noise_patterns:
        if re.search(pattern, token):  # Changed from match to search
            return None

    # Check for SentencePiece word boundary marker
    has_leading_space = token.startswith('▁')

    # Clean the token
    token = token.replace('▁', '')
    token = token.strip()

    # Filter garbage
    if len(token) == 0 or len(token) > 50:
        return None

    # Only keep tokens that look like actual words/punctuation
    if not re.search(r'[a-zA-Z0-9]', token):
        # No alphanumeric characters - probably garbage unless it's punctuation
        if token not in ['.', ',', '!', '?', ':', ';', '-', "'", '"']:
            return None

    return (token, has_leading_space)


def get_clean_words(tokens: List[str]) -> List[Tuple[str, List[int]]]:
    """
    Convert tokens to clean words, tracking which token indices make up each word.

    Returns
    -------
    words : list of (word, token_indices)
    """
    words = []
    current_word = ""
    current_indices = []

    for i, token in enumerate(tokens):
        result = clean_token(token)
        if result is None:
            continue

        cleaned, has_leading_space = result

        # If this starts a new word, save the previous one
        if has_leading_space and current_word:
            words.append((current_word, current_indices))
            current_word = cleaned
            current_indices = [i]
        elif has_leading_space and not current_word:
            # First word
            current_word = cleaned
            current_indices = [i]
        else:
            # Continue building current word (subword token)
            current_word += cleaned
            current_indices.append(i)

    # Don't forget the last word
    if current_word:
        words.append((current_word, current_indices))

    return words


def analyze_word_importance(
    attentions: np.ndarray,
    input_tokens: List[str],
    output_tokens: List[str]
) -> List[Tuple[str, float]]:
    """
    Calculate importance score for each input word.

    Returns
    -------
    word_scores : list of (word, importance_score) sorted by importance
    """
    # Get clean words from input tokens
    input_words = get_clean_words(input_tokens)

    if not input_words:
        return []

    # Use last layer attention (most refined)
    last_layer_attn = attentions[-1]  # (seq, seq)

    input_len = len(input_tokens)

    # For each input word, sum attention it receives from output tokens
    word_scores = []

    for word, token_indices in input_words:
        # Sum attention to all tokens in this word from all output positions
        total_attention = 0.0
        count = 0

        for out_pos in range(input_len, last_layer_attn.shape[0]):
            for token_idx in token_indices:
                if token_idx < last_layer_attn.shape[1]:
                    total_attention += last_layer_attn[out_pos, token_idx]
                    count += 1

        if count > 0:
            avg_attention = total_attention / count
            word_scores.append((word, float(avg_attention)))

    # Sort by importance
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return word_scores


def analyze_layer_stages(
    logits_per_layer: np.ndarray,
    vocab: List[str],
    num_stages: int = 3
) -> Dict:
    """
    Divide layers into stages and analyze what happens at each stage.

    Returns
    -------
    analysis : dict with keys:
        - stages: list of {name, layer_range, top_predictions, description}
        - final_answer: str
        - confidence: float
    """
    num_layers = logits_per_layer.shape[0]

    # Divide layers into stages
    stage_size = num_layers // num_stages
    stages = []

    stage_names = [
        "Input Understanding",
        "Knowledge Retrieval",
        "Answer Formation"
    ]

    for i in range(num_stages):
        start = i * stage_size
        end = (i + 1) * stage_size if i < num_stages - 1 else num_layers

        # Get predictions at the middle layer of this stage
        mid_layer = (start + end) // 2
        logits = logits_per_layer[mid_layer]

        # Softmax
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        # Get top predictions - try more to find clean tokens
        top_indices = np.argsort(probs)[-20:][::-1]  # Try top 20
        top_predictions = []

        for idx in top_indices:
            if len(top_predictions) >= 3:  # Stop after finding 3 clean ones
                break

            token = vocab[idx] if idx < len(vocab) else None
            if token:
                result = clean_token(token)
                if result:
                    cleaned, _ = result
                    if len(cleaned) > 0:
                        top_predictions.append({
                            "token": cleaned,
                            "probability": float(probs[idx])
                        })

        # Description based on stage
        if i == 0:
            desc = "Processing input tokens, identifying question type"
        elif i == 1:
            desc = "Retrieving relevant knowledge from training"
        else:
            desc = "Selecting and generating final answer"

        stages.append({
            "name": stage_names[i],
            "layer_range": (start, end - 1),
            "top_predictions": top_predictions,
            "description": desc
        })

    # Get final answer from last layer - try top predictions until we find a clean one
    final_logits = logits_per_layer[-1]
    final_probs = np.exp(final_logits - final_logits.max())
    final_probs /= final_probs.sum()

    # Try top 20 predictions to find clean tokens
    top_indices = np.argsort(final_probs)[-20:][::-1]
    final_answer = None
    final_confidence = 0.0
    alternatives = []

    for idx in top_indices:
        final_token = vocab[idx] if idx < len(vocab) else None
        if final_token:
            result = clean_token(final_token)
            if result:
                cleaned_token = result[0]
                prob = float(final_probs[idx])

                if final_answer is None:
                    # This is the top prediction
                    final_answer = cleaned_token
                    final_confidence = prob
                else:
                    # This is an alternative
                    alternatives.append({
                        "token": cleaned_token,
                        "probability": prob
                    })

                # Stop after finding 4 clean tokens (1 final + 3 alternatives)
                if len(alternatives) >= 3:
                    break

    # Fallback if nothing was clean
    if final_answer is None:
        final_answer = "(next token)"
        final_confidence = float(final_probs[top_indices[0]])

    return {
        "stages": stages,
        "final_answer": final_answer,
        "confidence": final_confidence,
        "alternatives": alternatives
    }


def create_answer_generation_flow(
    internals: dict,
    vocab: List[str]
) -> str:
    """
    Create a markdown visualization of how the answer was generated.

    Parameters
    ----------
    internals : dict
        Contains attentions, logits_per_layer, input_tokens, tokens
    vocab : list[str]
        Vocabulary

    Returns
    -------
    markdown : str
        Educational flow diagram
    """
    parts = []

    # Header
    parts.append("# 🎯 How the LLM Generated This Answer\n")

    # Get input question
    input_words = get_clean_words(internals["input_tokens"])
    question = " ".join([word for word, _ in input_words])

    # Get output answer
    output_words = get_clean_words(internals["tokens"])
    answer = " ".join([word for word, _ in output_words])

    parts.append(f"**Question**: _{question}_")
    parts.append(f"**Answer**: **{answer}**\n")
    parts.append("---\n")

    # Word importance
    parts.append("## 📊 Which Words Mattered Most?\n")
    word_scores = analyze_word_importance(
        internals["attentions"],
        internals["input_tokens"],
        internals["tokens"]
    )

    if word_scores:
        max_score = word_scores[0][1] if word_scores else 1.0

        for word, score in word_scores[:5]:  # Top 5
            if len(word.strip()) > 0:
                # Create bar
                bar_length = int((score / max_score) * 20)
                bar = "█" * bar_length
                percentage = int((score / max_score) * 100)

                parts.append(f"**{word}** {bar} {percentage}%\n")

        parts.append("💡 _The model paid most attention to these words when generating the answer_\n")

    parts.append("---\n")

    # Processing stages
    parts.append("## 🔄 Processing Pipeline\n")
    parts.append("_The model processes information in 3 stages:_\n")

    stage_analysis = analyze_layer_stages(
        internals["logits_per_layer"],
        vocab,
        num_stages=3
    )

    stage_emojis = ["🔍", "🧠", "✍️"]

    for i, stage in enumerate(stage_analysis["stages"]):
        emoji = stage_emojis[i]
        name = stage["name"]
        start, end = stage["layer_range"]
        desc = stage["description"]

        parts.append(f"### {emoji} Stage {i+1}: {name}")
        parts.append(f"_Layers {start}-{end}_\n")
        parts.append(f"{desc}\n")

        # Show top predictions at this stage - always show all available
        if stage["top_predictions"]:
            parts.append("**Leading predictions at this stage**:\n")
            for pred in stage["top_predictions"]:
                token = pred["token"]
                prob = pred["probability"]
                if prob >= 0.01:
                    parts.append(f"- `{token}` ({prob*100:.1f}%)\n")
                elif prob >= 0.001:
                    parts.append(f"- `{token}` ({prob*100:.2f}%)\n")
                else:
                    parts.append(f"- `{token}` ({prob*100:.3f}%)\n")
        else:
            parts.append("**Leading predictions at this stage**: _(no clear predictions)_\n")

        parts.append("")

    # Final answer
    parts.append("---\n")
    parts.append("## ✅ Final Answer\n")
    confidence = stage_analysis['confidence']

    # Format confidence nicely
    if confidence >= 0.01:
        conf_str = f"{confidence*100:.1f}%"
    else:
        conf_str = f"{confidence*100:.2f}%"

    parts.append(f"**{stage_analysis['final_answer']}** ({conf_str} confident)\n")

    if confidence > 0.5:
        parts.append("💪 _High confidence - the model is very sure_")
    elif confidence > 0.1:
        parts.append("🤔 _Medium confidence - some uncertainty_")
    elif confidence > 0.01:
        parts.append("❓ _Low confidence - uncertain_")
    else:
        parts.append("⚠️ _Very low confidence - highly uncertain (typical for large vocabularies)_")

    # Show alternatives that were considered
    if 'alternatives' in stage_analysis and stage_analysis['alternatives']:
        parts.append("\n")
        parts.append("**Alternatives considered**:\n")
        for alt in stage_analysis['alternatives']:
            prob = alt['probability']
            if prob >= 0.01:
                parts.append(f"- `{alt['token']}` ({prob*100:.1f}%)\n")
            elif prob >= 0.001:
                parts.append(f"- `{alt['token']}` ({prob*100:.2f}%)\n")
            else:
                parts.append(f"- `{alt['token']}` ({prob*100:.3f}%)\n")

    return "\n".join(parts)


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing answer generation flow...")

    # Mock internals
    num_layers = 22
    vocab_size = 32000

    input_tokens = ["<s>", "▁What", "▁is", "▁the", "▁capital", "▁of", "▁Australia", "?"]
    output_tokens = ["▁Can", "ber", "ra"]

    # Create vocab
    vocab = [""] * vocab_size
    vocab[100] = "▁Canberra"
    vocab[101] = "▁Sydney"
    vocab[102] = "▁Melbourne"
    for i in range(200, vocab_size):
        vocab[i] = f"token_{i}"

    # Mock logits - Simulate a confident model
    # Use small random noise, then boost correct answer strongly
    logits = np.random.randn(num_layers, vocab_size) * 0.3  # Small noise

    for l in range(num_layers):
        strength = l / num_layers  # Gets stronger with layers

        # Early layers: uncertain, multiple candidates
        if l < num_layers // 3:
            logits[l, 100] += 2.0  # Canberra (weak)
            logits[l, 101] += 2.5  # Sydney (initially stronger)
            logits[l, 102] += 1.8  # Melbourne
        # Middle layers: Canberra emerges
        elif l < 2 * num_layers // 3:
            logits[l, 100] += 5.0  # Canberra (getting stronger)
            logits[l, 101] += 3.0  # Sydney (fading)
            logits[l, 102] += 1.5  # Melbourne (fading)
        # Late layers: Canberra dominates (high confidence)
        else:
            logits[l, 100] += 10.0  # Canberra (very strong, ~10-20% after softmax)
            logits[l, 101] += 2.5   # Sydney (much weaker but still visible)
            logits[l, 102] += 1.8   # Melbourne (weak but still visible)

    # Mock attention
    seq_len = len(input_tokens) + len(output_tokens)
    attentions = np.random.rand(num_layers, seq_len, seq_len)
    attentions = attentions / attentions.sum(axis=2, keepdims=True)

    internals = {
        "input_tokens": input_tokens,
        "tokens": output_tokens,
        "attentions": attentions,
        "logits_per_layer": logits
    }

    flow = create_answer_generation_flow(internals, vocab)
    print(flow)

    print("\n✅ Test complete!")
