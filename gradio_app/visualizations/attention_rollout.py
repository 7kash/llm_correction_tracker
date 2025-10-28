"""
Attention Rollout Visualization

Shows how attention flows backward through layers to reveal which input tokens
contribute most to the final prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from typing import List, Optional


def compute_attention_rollout(
    attn_layers: np.ndarray,
    target_pos: int = -1,
    residual_alpha: float = 0.6
) -> np.ndarray:
    """
    Compute attention rollout from target position backward through layers.

    Parameters
    ----------
    attn_layers : np.ndarray
        Shape (num_layers, seq_len, seq_len), mean-over-heads attention
    target_pos : int
        Position to roll back from (-1 = last position)
    residual_alpha : float
        Residual mixing weight: A_hat = alpha*I + (1-alpha)*A

    Returns
    -------
    rollout : np.ndarray
        Shape (num_layers+1, seq_len) - contribution per token per stage
    """
    num_layers, seq_len, _ = attn_layers.shape

    if target_pos < 0:
        target_pos = seq_len + target_pos

    # Apply residual mixing and normalize
    mixed_attentions = []
    for A in attn_layers:
        A_hat = residual_alpha * np.eye(seq_len) + (1.0 - residual_alpha) * A
        # Row normalize
        A_hat = A_hat / (A_hat.sum(axis=1, keepdims=True) + 1e-12)
        mixed_attentions.append(A_hat)

    # Roll back from target position
    v_stages = []
    v = np.zeros(seq_len)
    v[target_pos] = 1.0
    v_stages.append(v.copy())

    # Propagate backward through layers
    for l in range(num_layers - 1, -1, -1):
        v = v @ mixed_attentions[l]  # Matrix multiply
        v_stages.append(v.copy())

    # Stack: index 0 = final, index L = input
    return np.stack(v_stages[::-1], axis=0)  # (L+1, seq_len)


def plot_attention_rollout(
    attn_layers: np.ndarray,
    tokens: List[str],
    target_pos: int = -1,
    top_k: int = 5,
    residual_alpha: float = 0.6,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Create attention rollout ribbon diagram.

    Parameters
    ----------
    attn_layers : np.ndarray
        Shape (num_layers, seq_len, seq_len)
    tokens : list[str]
        Token labels
    target_pos : int
        Position to analyze (-1 = last)
    top_k : int
        Number of top contributing tokens to show
    residual_alpha : float
        Residual mixing strength
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    seq_len = len(tokens)

    if target_pos < 0:
        target_pos = seq_len + target_pos

    # Compute rollout
    rollout = compute_attention_rollout(attn_layers, target_pos, residual_alpha)
    num_stages = rollout.shape[0]

    # Find top-k contributors
    scores = rollout.max(axis=0)
    top_idx = np.argsort(scores)[-top_k:][::-1]

    # Setup plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=0.0, vmax=rollout.max())

    x = np.arange(num_stages)
    lane_y = np.arange(seq_len) * 1.2
    lane_gap = 1.2

    # Draw ribbons for top-k tokens
    for idx in top_idx:
        contrib = rollout[:, idx]
        thick = contrib / (contrib.max() + 1e-12) * (0.6 * lane_gap)
        y_center = lane_y[idx]

        for i in range(num_stages - 1):
            val = 0.5 * (contrib[i] + contrib[i + 1])
            color = cmap(norm(float(val)))

            xl, xr = x[i], x[i + 1]
            yu_l = y_center + thick[i] / 2
            yl_l = y_center - thick[i] / 2
            yu_r = y_center + thick[i + 1] / 2
            yl_r = y_center - thick[i + 1] / 2

            xs = [xl, xr, xr, xl]
            ys = [yl_l, yl_r, yu_r, yu_l]
            ax.fill(xs, ys, color=color, alpha=0.9, linewidth=0)

        # Centerline
        ax.plot(x, np.full_like(x, y_center, dtype=float),
                linewidth=0.8, alpha=0.5, color='gray')

    # Token labels
    for i, tok in enumerate(tokens):
        txt = ax.text(x[0] - 0.15, lane_y[i], str(tok)[:15],
                      va='center', ha='right', fontsize=10, weight='bold')
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.95)])

    # Axis labels
    num_layers = num_stages - 1
    stage_labels = [f"L{num_layers - i}" for i in range(num_stages)]
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Layer (rollout from final → input)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"Attention Rollout — Position {target_pos} ({tokens[target_pos]})",
                 fontsize=13, weight='bold')

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Contribution', rotation=270, labelpad=15)

    # Grid lines
    for xi in x:
        ax.axvline(xi, linestyle='--', linewidth=0.5, alpha=0.2)

    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(-lane_gap, lane_y[-1] + lane_gap)

    plt.tight_layout()
    return fig


def get_top_contributors(
    attn_layers: np.ndarray,
    tokens: List[str],
    target_pos: int = -1,
    k: int = 5,
    residual_alpha: float = 0.6
) -> List[tuple]:
    """
    Get top-k contributing tokens with their scores.

    Returns
    -------
    list of (token, score) tuples
    """
    rollout = compute_attention_rollout(attn_layers, target_pos, residual_alpha)
    scores = rollout.max(axis=0)

    top_idx = np.argsort(scores)[-k:][::-1]
    return [(tokens[i], float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    num_layers = 8
    seq_len = 12
    tokens = ["Q:", "What", "is", "the", "capital", "of", "Australia", "?",
              "A:", "Can", "ber", "ra"]

    # Create synthetic attention
    attentions = []
    for _ in range(num_layers):
        raw = np.random.rand(seq_len, seq_len)
        raw = raw / raw.sum(axis=1, keepdims=True)
        attentions.append(raw)
    attentions = np.stack(attentions, axis=0)

    # Plot
    fig = plot_attention_rollout(attentions, tokens, target_pos=-1, top_k=5)
    fig.savefig("attention_rollout_test.png", dpi=150)
    print("✅ Saved attention_rollout_test.png")

    # Get top contributors
    top = get_top_contributors(attentions, tokens, target_pos=-1, k=5)
    print("\n🏆 Top 5 contributors:")
    for tok, score in top:
        print(f"  {tok}: {score:.3f}")
