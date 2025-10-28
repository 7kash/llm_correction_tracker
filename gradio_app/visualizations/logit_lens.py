"""
Logit Lens Visualization

Shows what the model "wants to say" at each intermediate layer by applying
the final LM head to hidden states.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax with temperature.

    Parameters
    ----------
    x : np.ndarray
        Logits
    temperature : float
        Softmax temperature (higher = more uniform)

    Returns
    -------
    probs : np.ndarray
        Probabilities
    """
    x_scaled = x / max(temperature, 1e-6)
    x_scaled = x_scaled - np.max(x_scaled, axis=-1, keepdims=True)
    exp_x = np.exp(x_scaled)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def get_top_k_per_layer(
    logits_per_layer: np.ndarray,
    vocab_tokens: List[str],
    k: int = 5,
    temperature: float = 1.0
) -> Tuple[List[List[str]], np.ndarray]:
    """
    Extract top-k tokens and probabilities per layer.

    Parameters
    ----------
    logits_per_layer : np.ndarray
        Shape (num_layers, vocab_size)
    vocab_tokens : list[str]
        Vocabulary tokens (for decoding)
    k : int
        Number of top tokens
    temperature : float
        Softmax temperature

    Returns
    -------
    tokens_per_layer : list[list[str]]
        Shape (num_layers, k)
    probs_per_layer : np.ndarray
        Shape (num_layers, k)
    """
    num_layers = logits_per_layer.shape[0]

    # Compute probabilities
    probs = softmax(logits_per_layer, temperature)  # (layers, vocab_size)

    tokens_per_layer = []
    probs_topk = []

    for layer_probs in probs:
        # Get top-k indices
        top_idx = np.argsort(layer_probs)[-k:][::-1]

        # Extract tokens and probs
        layer_tokens = [vocab_tokens[i] if i < len(vocab_tokens) else f"<{i}>"
                        for i in top_idx]
        layer_probs = layer_probs[top_idx]

        tokens_per_layer.append(layer_tokens)
        probs_topk.append(layer_probs)

    probs_topk = np.stack(probs_topk, axis=0)  # (layers, k)

    return tokens_per_layer, probs_topk


def plot_logit_lens_heatmap(
    tokens_per_layer: List[List[str]],
    probs_per_layer: np.ndarray,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot heatmap of top-k tokens per layer with labels.

    Parameters
    ----------
    tokens_per_layer : list[list[str]]
        Shape (num_layers, k)
    probs_per_layer : np.ndarray
        Shape (num_layers, k)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    num_layers, k = probs_per_layer.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Draw heatmap
    im = ax.imshow(probs_per_layer, aspect='auto', cmap='YlOrRd')

    # Overlay token labels
    for l in range(num_layers):
        for j in range(k):
            token = tokens_per_layer[l][j]
            prob = probs_per_layer[l, j]

            # Truncate long tokens
            if len(token) > 10:
                token = token[:8] + "..."

            # Choose text color based on background
            text_color = 'white' if prob > 0.5 else 'black'

            ax.text(j, l, f"{token}\n{prob:.2f}",
                    va='center', ha='center',
                    fontsize=9, color=text_color, weight='bold')

    # Axis labels
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Rank {i+1}" for i in range(k)])
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {l}" for l in range(num_layers)])

    ax.set_xlabel("Top-k Rank", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Logit Lens — Top-k Decoded Tokens per Layer",
                 fontsize=13, weight='bold')

    # Grid
    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_layers, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Probability', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig


def plot_token_probability_evolution(
    logits_per_layer: np.ndarray,
    vocab_tokens: List[str],
    tokens_to_track: List[str],
    temperature: float = 1.0,
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot probability evolution of specific tokens across layers.

    Parameters
    ----------
    logits_per_layer : np.ndarray
        Shape (num_layers, vocab_size)
    vocab_tokens : list[str]
        Vocabulary
    tokens_to_track : list[str]
        Tokens to plot
    temperature : float
        Softmax temperature
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    num_layers = logits_per_layer.shape[0]

    # Compute probabilities
    probs = softmax(logits_per_layer, temperature)  # (layers, vocab_size)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Find token indices
    token_indices = {}
    for token in tokens_to_track:
        if token in vocab_tokens:
            token_indices[token] = vocab_tokens.index(token)
        else:
            print(f"⚠️ Token '{token}' not found in vocabulary")

    # Plot trajectories
    x = np.arange(num_layers)
    for token, idx in token_indices.items():
        token_probs = probs[:, idx]
        ax.plot(x, token_probs, marker='o', linewidth=2.5,
                markersize=6, label=token, alpha=0.85)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Logit Lens — Token Probability Evolution",
                 fontsize=13, weight='bold')
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10, loc='best', frameon=True)

    plt.tight_layout()
    return fig


def analyze_layer_shift(
    logits_per_layer: np.ndarray,
    vocab_tokens: List[str],
    temperature: float = 1.0
) -> dict:
    """
    Analyze at which layer the top prediction changes.

    Parameters
    ----------
    logits_per_layer : np.ndarray
        Shape (num_layers, vocab_size)
    vocab_tokens : list[str]
        Vocabulary
    temperature : float
        Softmax temperature

    Returns
    -------
    analysis : dict
        Contains:
        - top_tokens_per_layer: list[str]
        - top_probs_per_layer: np.ndarray
        - shift_layers: list[int] (layers where top token changes)
    """
    probs = softmax(logits_per_layer, temperature)
    num_layers = probs.shape[0]

    top_tokens = []
    top_probs = []

    for layer_probs in probs:
        top_idx = np.argmax(layer_probs)
        top_tokens.append(vocab_tokens[top_idx] if top_idx < len(vocab_tokens) else f"<{top_idx}>")
        top_probs.append(float(layer_probs[top_idx]))

    # Find layers where top token changes
    shift_layers = []
    for l in range(1, num_layers):
        if top_tokens[l] != top_tokens[l - 1]:
            shift_layers.append(l)

    return {
        "top_tokens_per_layer": top_tokens,
        "top_probs_per_layer": np.array(top_probs),
        "shift_layers": shift_layers
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    num_layers = 12
    vocab_size = 100

    # Create vocabulary
    vocab = [f"token_{i}" for i in range(vocab_size)]
    vocab[10] = "Sydney"
    vocab[25] = "Canberra"
    vocab[30] = "Australia"

    # Create logits that shift from Sydney to Canberra
    logits_per_layer = []
    for l in range(num_layers):
        logits = np.random.randn(vocab_size) * 2

        # Early layers favor Sydney
        if l < 6:
            logits[10] += 3.0  # Sydney boost
        # Later layers favor Canberra
        else:
            logits[25] += 3.5  # Canberra boost

        logits_per_layer.append(logits)

    logits_per_layer = np.stack(logits_per_layer, axis=0)

    # Test top-k extraction
    tokens_per_layer, probs_per_layer = get_top_k_per_layer(
        logits_per_layer, vocab, k=5, temperature=1.0
    )

    # Plot heatmap
    fig1 = plot_logit_lens_heatmap(tokens_per_layer, probs_per_layer)
    fig1.savefig("logit_lens_heatmap_test.png", dpi=150)
    print("✅ Saved logit_lens_heatmap_test.png")

    # Plot evolution
    fig2 = plot_token_probability_evolution(
        logits_per_layer, vocab,
        tokens_to_track=["Sydney", "Canberra", "Australia"],
        temperature=1.0
    )
    fig2.savefig("logit_lens_evolution_test.png", dpi=150)
    print("✅ Saved logit_lens_evolution_test.png")

    # Analyze shifts
    analysis = analyze_layer_shift(logits_per_layer, vocab)
    print(f"\n🔄 Top token per layer:")
    for l, (tok, prob) in enumerate(zip(analysis['top_tokens_per_layer'],
                                         analysis['top_probs_per_layer'])):
        print(f"  L{l}: {tok} ({prob:.3f})")
    print(f"\n⚡ Prediction shifts at layers: {analysis['shift_layers']}")
