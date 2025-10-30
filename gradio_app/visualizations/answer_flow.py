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

    # Check for SentencePiece word boundary marker FIRST
    has_leading_space = token.startswith('▁')

    # Remove underscore prefix for further processing
    token = token.replace('▁', '')

    # AGGRESSIVE FILTERING: Remove common chat template words
    # TinyLlama uses tokens like "user", "assistant", "system" without brackets
    chat_keywords = ['user', 'assistant', 'system', 'im', 'start', 'end']
    token_lower = token.lower()
    if token_lower in chat_keywords:
        return None

    # Filter out tokens that are PART of chat keywords (like "assist" from "assistant")
    for keyword in chat_keywords:
        if keyword in token_lower and len(token_lower) <= len(keyword) + 2:
            return None

    # Remove any remaining chat template markers
    chat_markers = ['<|user|>', '<|assistant|>', '<|im_start|>', '<|im_end|>', '<|system|>', '|>']
    for marker in chat_markers:
        token = token.replace(marker, '')

    # Pattern-based filtering
    noise_patterns = [
        r'<[^>]*>',         # Anything in angle brackets
        r'^\s+$',           # Just whitespace
        r'^[<>|]+$',        # Just brackets and pipes
        r'^[▁]+$',          # Just underscores (redundant but safe)
    ]

    for pattern in noise_patterns:
        if re.search(pattern, token):
            return None

    token = token.strip()

    # Filter garbage
    if len(token) == 0 or len(token) > 50:
        return None

    # AGGRESSIVE: Only keep tokens with at least 2 letters OR known punctuation
    # This filters out random subword tokens like "ikz", "glob", "omer"
    if not re.search(r'[a-zA-Z]{2,}', token):
        # Less than 2 letters - only keep if it's known punctuation or single meaningful letter
        if token not in ['.', ',', '!', '?', ':', ';', '-', "'", '"', 'a', 'I', 'A']:
            return None

    # Filter out tokens that look like random fragments
    if len(token) >= 2:
        # Must have at least one vowel
        if not re.search(r'[aeiouAEIOU]', token):
            return None

        # Must be ASCII only (filters "ßen" and other non-English)
        if not token.isascii():
            return None

        # Filter tokens with 3+ consonants in a row at start (like "strk")
        if re.match(r'^[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{3,}', token):
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
        # Skip special tokens completely before processing
        if token in ['<s>', '</s>', '<pad>', '<unk>', '<|endoftext|>']:
            # If we have a word in progress, save it before skipping
            if current_word:
                words.append((current_word, current_indices))
                current_word = ""
                current_indices = []
            continue

        result = clean_token(token)
        if result is None:
            # Token was filtered out - save current word if any
            if current_word:
                words.append((current_word, current_indices))
                current_word = ""
                current_indices = []
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

    Parameters
    ----------
    logits_per_layer : np.ndarray
        Shape: (num_layers, vocab_size)
        Logits at each layer for the LAST generated token
    vocab : list[str]
        Vocabulary
    num_stages : int
        Number of stages to divide layers into

    Returns
    -------
    analysis : dict with keys:
        - stages: list of {name, layer_range, top_predictions, description}
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

    return {
        "stages": stages
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
        Contains attentions, logits_per_layer, input_tokens, tokens, token_alternatives
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

    # Show token-by-token generation with alternatives
    if "token_alternatives" in internals and internals["token_alternatives"]:
        parts.append("## 🔄 Generation Process\n")
        parts.append("_The model generated the answer token-by-token, considering alternatives at each step:_\n")

        token_alts = internals["token_alternatives"]

        for i, step in enumerate(token_alts[:10]):  # Show first 10 tokens max
            chosen = step["token"]
            chosen_prob = step["probability"]
            alternatives = step["alternatives"]

            # Clean the chosen token for display
            result = clean_token(chosen)
            if result:
                display_token, _ = result
            else:
                display_token = chosen.replace("▁", " ").strip()

            if not display_token:
                continue

            parts.append(f"### Token {i+1}: `{display_token}`\n")

            # Show top alternatives (excluding the chosen one)
            clean_alts = []
            for alt in alternatives[:5]:
                if alt["token"] != chosen:
                    alt_result = clean_token(alt["token"])
                    if alt_result:
                        alt_display, _ = alt_result
                        if alt_display and alt["probability"] >= 0.01:  # Only show >1%
                            clean_alts.append({
                                "token": alt_display,
                                "prob": alt["probability"]
                            })

            if clean_alts:
                parts.append(f"**Chose** `{display_token}` ({chosen_prob*100:.1f}%)\n")
                parts.append("**Alternatives considered**:\n")
                for alt in clean_alts[:3]:  # Show top 3
                    parts.append(f"- `{alt['token']}` ({alt['prob']*100:.1f}%)\n")
            else:
                parts.append(f"**Chose** `{display_token}` ({chosen_prob*100:.1f}% confident)\n")

            parts.append("")

        if len(token_alts) > 10:
            parts.append(f"_... and {len(token_alts) - 10} more tokens_\n")

        parts.append("---\n")

    # Word importance
    parts.append("## 📊 Which Words Mattered Most?\n")
    parts.append("_The model used attention to focus on these input words:_\n")

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

        parts.append("💡 _These words received the most attention during generation_\n")

    parts.append("---\n")

    # Show the final answer
    parts.append("## ✅ Final Answer\n")
    parts.append(f"**{answer}**\n")

    return "\n".join(parts)


def create_layer_by_layer_visualization(
    internals: dict,
    vocab: List[str]
) -> str:
    """
    Create visualization showing how answer forms through layers (logit lens).

    Parameters
    ----------
    internals : dict
        Contains layer_predictions, response, input_tokens
    vocab : list[str]
        Vocabulary (not used but kept for compatibility)

    Returns
    -------
    markdown : str
        Layer-by-layer visualization
    """
    parts = []

    # Get all input tokens and attention percentages
    input_tokens = internals.get("input_tokens", [])
    user_tokens = internals.get("user_question_tokens", [])
    attention_percentages = internals.get("attention_percentages", None)
    question = internals.get("question", "")
    context = internals.get("context", "")

    # Create whitelist: only words from actual question or context
    allowed_words = set()

    # Add words from question
    if question:
        question_words = question.lower().replace("?", " ").replace(".", " ").split()
        allowed_words.update(word.strip() for word in question_words if len(word.strip()) > 1)

    # Add words from context (correction)
    if context:
        context_words = context.lower().replace(":", " ").replace(".", " ").replace('"', " ").split()
        allowed_words.update(word.strip() for word in context_words if len(word.strip()) > 1)

    # Get clean words from all input
    all_words = get_clean_words(input_tokens)

    # Filter to only words in the whitelist (actual question/context)
    filtered_words = []
    for word, token_indices in all_words:
        word_lower = word.lower().strip()

        # Skip punctuation-only
        if all(c in '.,!?;:\'"()-' for c in word):
            continue

        # Only include if in whitelist
        if word_lower in allowed_words:
            filtered_words.append((word, token_indices))

    # Attention Section - Theory then Data
    parts.append("## 📚 Theory: Attention Mechanism\n\n")
    parts.append("_The model uses **attention** to decide which input words are most relevant for generating the answer._\n\n")
    parts.append("**Mathematical Formula:**\n\n")
    parts.append("$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$\n\n")
    parts.append("_Each word gets an attention score (0-100%) showing how much the model \"focused\" on it. ")
    parts.append("Higher scores mean the word had more influence on the answer._\n\n")

    # Add attention visualization (chart)
    parts.append("### Attention Distribution\n\n")

    if filtered_words and attention_percentages:
        # Map tokens to words and sum their attention
        word_attention = {}

        for word, token_indices in filtered_words:
            total_attention = 0.0
            for idx in token_indices:
                if idx < len(attention_percentages):
                    total_attention += attention_percentages[idx]

            if total_attention > 0:
                word_attention[word] = total_attention

        if word_attention:
            # Sort by attention (highest first)
            sorted_words = sorted(word_attention.items(), key=lambda x: x[1], reverse=True)[:8]

            # Create minimalistic HTML chart with REAL percentages
            parts.append('<div style="margin: 1.5rem 0;">\n')
            for word, attention_pct in sorted_words:
                # Use real percentage (attention_percentages already sum to 100)
                # Display percentage rounded to 1 decimal
                display_pct = round(attention_pct, 1)
                # Bar width is the actual percentage
                bar_width = min(100, int(attention_pct))

                parts.append(f'<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">\n')
                parts.append(f'  <span style="min-width: 100px; text-align: right; font-size: 0.875rem; color: #374151; font-weight: 500;">{word}</span>\n')
                parts.append(f'  <div style="flex: 1; background: #E5E7EB; height: 6px; border-radius: 3px; overflow: hidden;">\n')
                parts.append(f'    <div style="width: {bar_width}%; height: 100%; background: #111827;"></div>\n')
                parts.append(f'  </div>\n')
                parts.append(f'  <span style="min-width: 45px; font-size: 0.75rem; color: #6B7280;">{display_pct}%</span>\n')
                parts.append(f'</div>\n')
            parts.append('</div>\n\n')
            parts.append("_Real attention percentages from the model's final layer (sum to 100%)_\n\n")
        else:
            parts.append("_No significant words found._\n\n")
    else:
        parts.append("_Attention data not available._\n\n")

    parts.append("---\n\n")

    # Softmax Section - Theory then Data
    softmax_example = internals.get("softmax_example", None)

    if softmax_example:
        parts.append("## 📚 Theory: Softmax Transformation\n\n")
        parts.append("_The model outputs **logits** (raw scores) for each possible next token. ")
        parts.append("Softmax transforms these into probabilities (0-100%) that sum to 100%._\n\n")
        parts.append("**Mathematical Formula:**\n\n")
        parts.append("$$\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$\n\n")
        parts.append("_The exponential amplifies differences between scores._\n\n")

        # Create softmax visualization (chart)
        parts.append("### Top Token Probabilities\n\n")

        parts.append('<div style="margin: 1.5rem 0;">\n')
        for item in softmax_example:
            token = item["token"]
            logit = item["logit"]
            prob = item["probability"]
            # Use REAL probability percentage (not normalized)
            prob_pct = prob * 100
            bar_width = min(100, int(prob_pct))

            parts.append(f'<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem; padding: 0.5rem 0;">\n')
            parts.append(f'  <span style="min-width: 90px; text-align: right; font-size: 0.875rem; color: #374151; font-weight: 500;">{token}</span>\n')
            parts.append(f'  <span style="min-width: 70px; font-size: 0.75rem; color: #6B7280;">logit: {logit:+.2f}</span>\n')
            parts.append(f'  <div style="flex: 1; background: #E5E7EB; height: 4px; border-radius: 2px; overflow: hidden;">\n')
            parts.append(f'    <div style="width: {bar_width}%; height: 100%; background: #111827;"></div>\n')
            parts.append(f'  </div>\n')
            parts.append(f'  <span style="min-width: 55px; font-size: 0.75rem; color: #374151; font-weight: 500;">{prob_pct:.1f}%</span>\n')
            parts.append(f'</div>\n')
        parts.append('</div>\n\n')
        parts.append("_Real softmax probabilities from the model (these top 5 tokens sum to less than 100% as other tokens have remaining probability)_\n\n")
        parts.append("---\n\n")

    # Theoretical explanation for logit lens
    parts.append("## 📚 Theory: Logit Lens (Layer-by-Layer Predictions)\n")
    parts.append("_The **logit lens** technique reveals what the model \"thinks\" at each layer. ")
    parts.append("Normally, only the final layer produces the answer. ")
    parts.append("But we can apply the final prediction head to ANY layer to see what it would predict at that point. ")
    parts.append("Early layers are uncertain (low confidence), while later layers refine and become confident._\n\n")
    parts.append("---\n\n")

    parts.append("## 🎯 Layer-by-Layer Predictions\n\n")

    layer_predictions = internals["layer_predictions"]

    # Show every 3rd layer to avoid clutter (or all if few layers)
    num_layers = len(layer_predictions)
    if num_layers > 15:
        # Show layers: 0, 3, 6, 9, ..., last
        layer_indices = list(range(0, num_layers, 3)) + [num_layers - 1]
    else:
        layer_indices = range(num_layers)

    for layer_idx in layer_indices:
        if layer_idx >= len(layer_predictions):
            continue

        layer_data = layer_predictions[layer_idx]
        predictions = layer_data["predictions"]

        # Separate actual answer from alternatives
        actual_answer_pred = None
        alternatives = []

        for pred in predictions:
            token = pred["token"]
            prob = pred["probability"]
            is_actual = pred.get("is_actual_answer", False)

            # Clean token for display
            result = clean_token(token)
            if result:
                display_token, _ = result
            else:
                display_token = token.strip()

            if not display_token:
                display_token = token.strip()  # Use raw if cleaning failed

            pred_data = {"token": display_token, "prob": prob}

            if is_actual:
                actual_answer_pred = pred_data
            else:
                alternatives.append(pred_data)

        # Show layer number
        parts.append(f"### Layer {layer_idx}\n")

        # ALWAYS show actual answer first
        if actual_answer_pred:
            token = actual_answer_pred["token"]
            prob = actual_answer_pred["prob"]
            parts.append(f"**Actual answer** `{token}`: {prob*100:.1f}%\n")

        # Then show top 3 alternatives
        if alternatives:
            parts.append("**Top alternatives**:\n")
            for alt in alternatives[:3]:
                parts.append(f"- `{alt['token']}` ({alt['prob']*100:.1f}%)\n")

        parts.append("\n")

    parts.append("---\n\n")

    # Theoretical explanation for final answer
    parts.append("## 📚 Theory: How the Final Answer is Selected\n")
    parts.append("_The model generates text token-by-token. At each position, the final layer produces a ")
    parts.append("**probability distribution** over all possible tokens. The model selects the token with highest probability ")
    parts.append("(greedy decoding). This process repeats until a stop condition is met._\n\n")
    parts.append("---\n\n")

    parts.append("## ✅ Final Answer\n\n")
    parts.append(f"**{internals['response']}**\n\n")
    parts.append("_This is what the final layer predicted with highest confidence!_\n")

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

    # Mock token alternatives (simulating token-by-token generation)
    token_alternatives = [
        {
            "token": "▁Can",
            "probability": 0.45,
            "alternatives": [
                {"token": "▁Can", "probability": 0.45},
                {"token": "▁Sydney", "probability": 0.12},
                {"token": "▁Melbourne", "probability": 0.08},
                {"token": "▁The", "probability": 0.05},
            ]
        },
        {
            "token": "ber",
            "probability": 0.89,
            "alternatives": [
                {"token": "ber", "probability": 0.89},
                {"token": "berra", "probability": 0.03},
            ]
        },
        {
            "token": "ra",
            "probability": 0.92,
            "alternatives": [
                {"token": "ra", "probability": 0.92},
            ]
        }
    ]

    internals = {
        "input_tokens": input_tokens,
        "tokens": output_tokens,
        "token_alternatives": token_alternatives,
        "attentions": attentions,
        "logits_per_layer": logits
    }

    flow = create_answer_generation_flow(internals, vocab)
    print(flow)

    print("\n" + "="*60)
    print("Testing layer-by-layer visualization...")
    print("="*60 + "\n")

    # Mock layer predictions (logit lens) - NEW FORMAT with is_actual_answer flag
    layer_predictions = []
    for layer in range(num_layers):
        if layer < 7:
            # Early layers: Canberra has low probability, alternatives higher
            preds = [
                {"token": "Canberra", "probability": 0.05, "is_actual_answer": True},  # Actual answer, low prob
                {"token": "▁Sydney", "probability": 0.15, "is_actual_answer": False},
                {"token": "▁Melbourne", "probability": 0.12, "is_actual_answer": False},
                {"token": "▁The", "probability": 0.08, "is_actual_answer": False},
            ]
        elif layer < 15:
            # Middle layers: Canberra climbing but not yet top
            preds = [
                {"token": "Canberra", "probability": 0.25, "is_actual_answer": True},  # Getting stronger
                {"token": "▁Sydney", "probability": 0.18, "is_actual_answer": False},
                {"token": "▁Melbourne", "probability": 0.10, "is_actual_answer": False},
            ]
        else:
            # Late layers: Canberra dominates
            preds = [
                {"token": "Canberra", "probability": 0.65, "is_actual_answer": True},  # Confident!
                {"token": "▁Sydney", "probability": 0.08, "is_actual_answer": False},
                {"token": "▁Melbourne", "probability": 0.03, "is_actual_answer": False},
            ]

        layer_predictions.append({
            "layer": layer,
            "predictions": preds
        })

    layer_internals = {
        "input_tokens": input_tokens,
        "response": "Canberra",
        "layer_predictions": layer_predictions
    }

    layer_viz = create_layer_by_layer_visualization(layer_internals, vocab)
    print(layer_viz)

    print("\n✅ All tests complete!")
