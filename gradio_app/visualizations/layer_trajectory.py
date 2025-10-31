"""
Layer Trajectory Visualization

Tracks how token representations evolve through layers in 2D space (PCA).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def pca_2d(X: np.ndarray) -> tuple:
    """
    Compute 2D PCA projection.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)

    Returns
    -------
    X_2d : np.ndarray
        Shape (n_samples, 2)
    mean : np.ndarray
        Mean vector for centering
    components : np.ndarray
        Principal components (2, n_features)
    """
    # Center data
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project to 2D
    X_2d = X_centered @ Vt[:2].T

    return X_2d, mean, Vt[:2]


def plot_layer_trajectory(
    hidden_states: np.ndarray,
    token_name: str,
    figsize: tuple = (8, 6)
) -> plt.Figure:
    """
    Plot trajectory of a single token through layers.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (num_layers, hidden_dim)
    token_name : str
        Name of the token being tracked
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    num_layers, hidden_dim = hidden_states.shape

    # 2D PCA
    coords_2d, _, _ = pca_2d(hidden_states)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Trajectory line
    ax.plot(coords_2d[:, 0], coords_2d[:, 1],
            linewidth=2.5, marker='o', markersize=8,
            color='#1f77b4', alpha=0.8, label=token_name)

    # Annotate layers
    for l in [0, num_layers // 2, num_layers - 1]:
        ax.annotate(f"L{l}", coords_2d[l],
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Axis labels
    ax.set_xlabel("PC1 (Principal Component 1)", fontsize=11)
    ax.set_ylabel("PC2 (Principal Component 2)", fontsize=11)
    ax.set_title(f"Layer Trajectory: {token_name}", fontsize=13, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def plot_multi_token_trajectories(
    hidden_states_dict: Dict[str, np.ndarray],
    figsize: tuple = (10, 7)
) -> plt.Figure:
    """
    Plot trajectories for multiple tokens on shared PCA space.

    Parameters
    ----------
    hidden_states_dict : dict
        Maps token_name -> hidden_states array (num_layers, hidden_dim)
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    # Collect all hidden states
    all_hidden = []
    token_names = []
    layer_counts = []

    for token_name, hidden in hidden_states_dict.items():
        all_hidden.append(hidden)
        token_names.append(token_name)
        layer_counts.append(hidden.shape[0])

    # Stack and compute shared PCA
    all_hidden_stacked = np.vstack(all_hidden)
    all_coords_2d, _, _ = pca_2d(all_hidden_stacked)

    # Split back by token
    coords_by_token = {}
    start_idx = 0
    for token_name, num_layers in zip(token_names, layer_counts):
        coords_by_token[token_name] = all_coords_2d[start_idx:start_idx + num_layers]
        start_idx += num_layers

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(token_names)))

    for i, token_name in enumerate(token_names):
        coords = coords_by_token[token_name]
        num_layers = coords.shape[0]

        ax.plot(coords[:, 0], coords[:, 1],
                linewidth=2.5, marker='o', markersize=7,
                color=colors[i], alpha=0.85, label=token_name)

        # Annotate first and last layer
        ax.annotate("L0", coords[0],
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color=colors[i])
        ax.annotate(f"L{num_layers-1}", coords[-1],
                    xytext=(5, -10), textcoords="offset points",
                    fontsize=9, color=colors[i])

    ax.set_xlabel("PC1 (Principal Component 1)", fontsize=11)
    ax.set_ylabel("PC2 (Principal Component 2)", fontsize=11)
    ax.set_title("Layer Trajectories (Shared PCA Space)", fontsize=13, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10, loc='best', frameon=True)

    plt.tight_layout()
    return fig


def compare_trajectories_before_after(
    hidden_before: np.ndarray,
    hidden_after: np.ndarray,
    token_name: str,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Compare trajectory before and after correction on shared PCA space.

    Parameters
    ----------
    hidden_before : np.ndarray
        Shape (num_layers, hidden_dim) before correction
    hidden_after : np.ndarray
        Shape (num_layers, hidden_dim) after correction
    token_name : str
        Token being tracked
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    # Compute shared PCA
    combined = np.vstack([hidden_before, hidden_after])
    coords_2d, _, _ = pca_2d(combined)

    num_layers = hidden_before.shape[0]
    coords_before = coords_2d[:num_layers]
    coords_after = coords_2d[num_layers:]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Before correction
    ax.plot(coords_before[:, 0], coords_before[:, 1],
            linewidth=2.5, marker='o', markersize=7,
            color='#ff7f0e', alpha=0.85, label=f"Before correction")

    # After correction
    ax.plot(coords_after[:, 0], coords_after[:, 1],
            linewidth=2.5, marker='s', markersize=7,
            color='#2ca02c', alpha=0.85, label=f"After correction")

    # Annotate divergence point (find where trajectories differ most)
    distances = np.linalg.norm(coords_before - coords_after, axis=1)
    diverge_layer = np.argmax(distances)

    ax.scatter([coords_before[diverge_layer, 0]],
               [coords_before[diverge_layer, 1]],
               s=200, facecolors='none', edgecolors='red',
               linewidth=2.5, label=f"Divergence (L{diverge_layer})")

    # Annotations
    ax.annotate("L0", coords_before[0],
                xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.annotate(f"L{num_layers-1}", coords_before[-1],
                xytext=(5, -10), textcoords="offset points", fontsize=9)

    ax.set_xlabel("PC1 (Principal Component 1)", fontsize=11)
    ax.set_ylabel("PC2 (Principal Component 2)", fontsize=11)
    ax.set_title(f"Trajectory Comparison: {token_name}\n(Before vs After Correction)",
                 fontsize=13, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10, loc='best', frameon=True)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    num_layers = 12
    hidden_dim = 64

    def make_trajectory(layers, dim, drift=0.5):
        """Create synthetic trajectory with momentum."""
        x = np.random.randn(dim)
        traj = [x.copy()]
        v = np.random.randn(dim) * 0.2
        for _ in range(layers - 1):
            v = 0.8 * v + 0.2 * np.random.randn(dim) * drift
            x = x + v
            traj.append(x.copy())
        return np.stack(traj, axis=0)

    # Single token
    hidden = make_trajectory(num_layers, hidden_dim)
    fig1 = plot_layer_trajectory(hidden, "Australia")
    fig1.savefig("trajectory_single_test.png", dpi=150)
    print("✅ Saved trajectory_single_test.png")

    # Multiple tokens
    hidden_dict = {
        "Australia": make_trajectory(num_layers, hidden_dim),
        "capital": make_trajectory(num_layers, hidden_dim),
        "Canberra": make_trajectory(num_layers, hidden_dim)
    }
    fig2 = plot_multi_token_trajectories(hidden_dict)
    fig2.savefig("trajectory_multi_test.png", dpi=150)
    print("✅ Saved trajectory_multi_test.png")

    # Before/after comparison
    hidden_before = make_trajectory(num_layers, hidden_dim, drift=0.4)
    hidden_after = make_trajectory(num_layers, hidden_dim, drift=0.6)
    fig3 = compare_trajectories_before_after(hidden_before, hidden_after, "capital")
    fig3.savefig("trajectory_comparison_test.png", dpi=150)
    print("✅ Saved trajectory_comparison_test.png")
