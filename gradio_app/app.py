"""
LLM Inference Tracker - Gradio App

Track how LLMs change their responses when corrected, with visualizations
of internal mechanisms: attention, layer trajectories, and logit lens.

Deployment: Hugging Face Spaces
"""

import gradio as gr
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from backend.llm_with_internals import LLMWithInternals
from visualizations.attention_rollout import (
    plot_attention_rollout,
    get_top_contributors
)
from visualizations.layer_trajectory import (
    plot_layer_trajectory,
    compare_trajectories_before_after
)
from visualizations.logit_lens import (
    plot_logit_lens_heatmap,
    plot_token_probability_evolution,
    get_top_k_per_layer,
    analyze_layer_shift
)
from visualizations.answer_flow import (
    create_answer_generation_flow,
    analyze_word_importance
)


# Global model instance (loaded once)
MODEL = None
SESSION_HISTORY = []  # List of (question, response, internals) tuples


def load_model():
    """Load model on first use (lazy loading)."""
    global MODEL
    if MODEL is None:
        print("🔧 Loading TinyLlama model...")
        MODEL = LLMWithInternals(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("✅ Model ready!")
    return MODEL


def get_vocab_list():
    """
    Get vocabulary as a list indexed by token ID.

    Returns
    -------
    list[str] where vocab[token_id] = token_string
    """
    vocab_dict = MODEL.tokenizer.get_vocab()
    vocab_size = len(vocab_dict)
    vocab = [""] * vocab_size
    for token_str, token_id in vocab_dict.items():
        vocab[token_id] = token_str
    return vocab


def generate_response(
    question: str,
    max_tokens: int = 50,
    temperature: float = 0.7
) -> Tuple[str, dict]:
    """
    Generate response with internals extraction.

    Returns
    -------
    response_text : str
    internals : dict with attention, hidden_states, logits, etc.
    """
    model = load_model()

    # Build history from session
    history = [(q, r) for q, r, _ in SESSION_HISTORY] if SESSION_HISTORY else None

    # Generate with internals
    result = model.generate_with_internals(
        question=question,
        history=history,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )

    # Store in session
    SESSION_HISTORY.append((question, result["response"], result))

    return result["response"], result


def create_turn_summary(internals: dict, previous_internals: Optional[dict] = None) -> str:
    """
    Create answer generation flow visualization.

    Returns Markdown-formatted text.
    """
    vocab = get_vocab_list()
    return create_answer_generation_flow(internals, vocab)


def visualize_attention(turn_index: int) -> plt.Figure:
    """Create attention rollout visualization for a specific turn."""
    if turn_index >= len(SESSION_HISTORY):
        return None

    _, _, internals = SESSION_HISTORY[turn_index]

    # Get tokens and attention
    all_tokens = internals["input_tokens"] + internals["tokens"]
    attentions = internals["attentions"]

    # Create visualization
    fig = plot_attention_rollout(
        attentions,
        tokens=all_tokens,
        target_pos=-1,
        top_k=min(7, len(all_tokens)),
        residual_alpha=0.6
    )

    return fig


def visualize_trajectories(turn_index: int, token_to_track: Optional[str] = None) -> plt.Figure:
    """Create layer trajectory visualization."""
    if turn_index >= len(SESSION_HISTORY):
        return None

    _, _, internals = SESSION_HISTORY[turn_index]

    # If comparing with previous turn
    if turn_index > 0 and token_to_track:
        _, _, prev_internals = SESSION_HISTORY[turn_index - 1]

        # For simplicity, use the final token's hidden states
        fig = compare_trajectories_before_after(
            hidden_before=prev_internals["hidden_states"],
            hidden_after=internals["hidden_states"],
            token_name=token_to_track or "last_token"
        )
    else:
        # Single trajectory
        fig = plot_layer_trajectory(
            internals["hidden_states"],
            token_name=token_to_track or "last_token"
        )

    return fig


def visualize_logit_lens(turn_index: int, mode: str = "heatmap") -> plt.Figure:
    """Create logit lens visualization."""
    if turn_index >= len(SESSION_HISTORY):
        return None

    _, _, internals = SESSION_HISTORY[turn_index]

    logits = internals["logits_per_layer"]
    vocab = get_vocab_list()

    if mode == "heatmap":
        tokens_per_layer, probs_per_layer = get_top_k_per_layer(
            logits, vocab, k=5, temperature=1.0
        )
        fig = plot_logit_lens_heatmap(tokens_per_layer, probs_per_layer)

    elif mode == "evolution":
        # Track specific tokens (for now, just show top 3 from final layer)
        final_probs = np.exp(logits[-1] - logits[-1].max())
        final_probs /= final_probs.sum()
        top_idx = np.argsort(final_probs)[-3:][::-1]
        tokens_to_track = [vocab[i] for i in top_idx]

        fig = plot_token_probability_evolution(
            logits, vocab, tokens_to_track, temperature=1.0
        )

    return fig


def reset_session():
    """Clear session history."""
    global SESSION_HISTORY
    SESSION_HISTORY = []
    return "Session reset! Start with a new question.", None, None, None, None


# ============================================================================
# Gradio Interface
# ============================================================================

def main_interface():
    with gr.Blocks(title="🧠 LLM Inference Tracker", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🧠 LLM Inference Tracker

        **Track how LLMs change their responses when corrected** — with visualizations of:
        - 🎯 **Attention Rollout**: Which words the model focuses on
        - 📈 **Layer Trajectories**: How representations evolve through layers
        - 🔍 **Logit Lens**: What the model "wants to say" at each layer

        **Model**: TinyLlama-1.1B (runs locally, ~2-5 seconds per response)

        ---
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions & Provide Corrections")

                question_input = gr.Textbox(
                    label="Question or Correction",
                    placeholder="Example: What is the capital of Australia?",
                    lines=3
                )

                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate Response", variant="primary")
                    reset_btn = gr.Button("🔄 Reset Session", variant="secondary")

                response_output = gr.Textbox(
                    label="Model Response",
                    lines=5,
                    interactive=False
                )

                turn_summary = gr.Markdown("_No turns yet. Ask a question to start!_")

            with gr.Column(scale=1):
                gr.Markdown("### ℹ️ Quick Guide")
                gr.Markdown("""
                **How to use**:
                1. Ask a question
                2. Get response + internals extracted
                3. Provide correction (optional)
                4. Compare visualizations

                **Tips**:
                - Use simple questions for best results
                - Compare before/after corrections
                - Try: "What is the capital of Australia?" → "Actually it's Canberra"
                """)

        gr.Markdown("---")
        gr.Markdown("## 📊 Visualizations")

        with gr.Tab("🎯 Attention Rollout"):
            gr.Markdown("""
            **Shows**: Which input tokens the model paid attention to when generating the response.

            **How to read**: Thicker ribbons = more attention. Colors show contribution strength.
            """)
            turn_selector_attn = gr.Slider(
                minimum=0, maximum=10, step=1, value=0,
                label="Select Turn to Visualize"
            )
            attention_plot = gr.Plot(label="Attention Rollout")
            visualize_attn_btn = gr.Button("🔍 Show Attention")

        with gr.Tab("📈 Layer Trajectories"):
            gr.Markdown("""
            **Shows**: How hidden representations evolve from layer 0 to final layer.

            **How to read**: Lines show the path through 2D PCA space. Divergence = where correction takes effect.
            """)
            turn_selector_traj = gr.Slider(
                minimum=0, maximum=10, step=1, value=0,
                label="Select Turn to Visualize"
            )
            trajectory_plot = gr.Plot(label="Layer Trajectory")
            visualize_traj_btn = gr.Button("🔍 Show Trajectory")

        with gr.Tab("🔍 Logit Lens"):
            gr.Markdown("""
            **Shows**: What the model "wants to say" at each intermediate layer.

            **How to read**: Heatmap shows top-k tokens per layer. See when prediction changes!
            """)
            turn_selector_logit = gr.Slider(
                minimum=0, maximum=10, step=1, value=0,
                label="Select Turn to Visualize"
            )
            logit_mode = gr.Radio(
                choices=["heatmap", "evolution"],
                value="heatmap",
                label="Visualization Mode"
            )
            logit_plot = gr.Plot(label="Logit Lens")
            visualize_logit_btn = gr.Button("🔍 Show Logit Lens")

        # Event handlers
        def on_generate(question):
            if not question.strip():
                return "Please enter a question!", None, None, None, None

            response, internals = generate_response(question)

            # Create summary
            prev_internals = SESSION_HISTORY[-2][2] if len(SESSION_HISTORY) > 1 else None
            summary = create_turn_summary(internals, prev_internals)

            return response, summary, None, None, None

        generate_btn.click(
            fn=on_generate,
            inputs=[question_input],
            outputs=[response_output, turn_summary, attention_plot, trajectory_plot, logit_plot]
        )

        reset_btn.click(
            fn=reset_session,
            inputs=[],
            outputs=[turn_summary, attention_plot, trajectory_plot, logit_plot, response_output]
        )

        visualize_attn_btn.click(
            fn=visualize_attention,
            inputs=[turn_selector_attn],
            outputs=[attention_plot]
        )

        visualize_traj_btn.click(
            fn=visualize_trajectories,
            inputs=[turn_selector_traj],
            outputs=[trajectory_plot]
        )

        visualize_logit_btn.click(
            fn=visualize_logit_lens,
            inputs=[turn_selector_logit, logit_mode],
            outputs=[logit_plot]
        )

    return demo


if __name__ == "__main__":
    demo = main_interface()
    demo.queue()  # Enable queuing for concurrent users
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,  # HF Spaces default
        share=False
    )
