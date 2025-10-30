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
    create_layer_by_layer_visualization,
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


def is_one_word_question(question: str) -> Tuple[bool, str]:
    """
    Check if question is likely answerable in one word.

    Returns
    -------
    (is_valid, message) : tuple
        is_valid: True if question looks one-word answerable
        message: Explanation if not valid
    """
    question_lower = question.lower().strip()

    # Check for common one-word question patterns
    one_word_patterns = [
        'what is',
        'who is',
        'where is',
        'when was',
        'when did',
        'which',
        'what was',
        'capital of',
        'color of',
        'color is',
        'president of',
        'author of',
        'founder of'
    ]

    # Check for multi-word answer indicators
    multi_word_indicators = [
        'why',
        'how does',
        'explain',
        'describe',
        'tell me about',
        'what are the'
    ]

    # If contains multi-word indicators, reject
    for indicator in multi_word_indicators:
        if indicator in question_lower:
            return False, f"Questions with '{indicator}' usually need more than one word. Try: 'What is the capital of Australia?'"

    # If contains one-word patterns, accept
    for pattern in one_word_patterns:
        if pattern in question_lower:
            return True, ""

    # Default: accept but warn
    return True, "Note: This works best with questions that have one-word answers (like 'What is X?' or 'Who is Y?')"


def generate_one_word_answer(question: str) -> Tuple[str, str]:
    """
    Generate one-word answer and show layer-by-layer predictions (logit lens).

    Returns
    -------
    answer : str
    visualization : str (markdown)
    """
    # Validate question
    is_valid, message = is_one_word_question(question)

    if not is_valid:
        error_viz = f"## ⚠️ Question Not Suitable\n\n{message}\n\n**Examples of good questions:**\n- What is the capital of Australia?\n- Who is the president of USA?\n- What color is the sky?\n"
        return "❌ Please ask a one-word answerable question", error_viz

    model = load_model()

    # Generate with layer-by-layer tracking
    result = model.generate_one_word_with_layers(question, max_new_tokens=10)

    # Check if answer is actually one word (3 tokens max for compound words)
    answer_word_count = len(result["response"].split())
    if answer_word_count > 2:
        warning_viz = f"## ⚠️ Answer Too Long\n\nThe model generated: **{result['response']}** ({answer_word_count} words)\n\nThis visualization works best with single-word answers. Try rephrasing your question to expect a one-word answer.\n\n**Good examples:**\n- What is the capital of Australia?\n- Who discovered gravity?\n- What color is grass?"
        return result["response"], warning_viz

    # Create visualization
    vocab = get_vocab_list()
    visualization = create_layer_by_layer_visualization(result, vocab)

    # Add helpful note if there was a warning
    if message:
        visualization = f"💡 {message}\n\n---\n\n{visualization}"

    return result["response"], visualization


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
    return {
        "response": "",
        "summary": "_Ask a question to see how the model forms its answer through layers!_",
        "correction_input": gr.update(visible=False, value=""),
        "correction_btn": gr.update(visible=False),
        "comparison": gr.update(visible=False, value="")
    }


# ============================================================================
# Gradio Interface
# ============================================================================

def main_interface():
    with gr.Blocks(title="🧠 LLM Inference Tracker", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🧠 LLM Layer-by-Layer Visualization

        **See how a language model forms its answer through all 22 layers!**

        Ask a one-word answerable question and watch the model's prediction evolve from uncertain (early layers) to confident (final layers).

        ### What You'll See:
        - 🔬 **Layer-by-Layer Predictions**: The answer at each of 22 layers with probabilities
        - 🎯 **Top Alternatives**: Other answers the model considered
        - 📊 **Confidence Evolution**: How certainty builds through layers
        - 🔄 **Before/After Corrections**: Compare how corrections affect layer predictions

        **Model**: TinyLlama-1.1B (runs locally, ~2-5 seconds per response)

        ---
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask a Question")
                gr.Markdown("_Ask questions that can be answered in one word (e.g., 'What is the capital of Australia?')_")

                question_input = gr.Textbox(
                    label="Question (expecting one-word answer)",
                    placeholder="Example: What is the capital of Australia?",
                    lines=2
                )

                generate_btn = gr.Button("🧠 Generate Answer & Show Layers", variant="primary", size="lg")

                response_output = gr.Textbox(
                    label="Model's Answer",
                    lines=2,
                    interactive=False
                )

                gr.Markdown("---")

                gr.Markdown("### 🔄 Provide a Correction (Optional)")
                gr.Markdown("_If the answer is wrong, provide a correction to see how the model adapts._")

                correction_input = gr.Textbox(
                    label="Correction (provide alternative answer or new context)",
                    placeholder="Example: Green (for unripe banana) OR What color is unripe banana?",
                    lines=2,
                    visible=False  # Only show after first answer
                )

                with gr.Row():
                    correction_btn = gr.Button("📝 Submit Correction", variant="secondary", visible=False)
                    reset_btn = gr.Button("🔄 Reset Session", variant="secondary")

                comparison_output = gr.Markdown("", visible=False)  # For showing before/after

                turn_summary = gr.Markdown("_Ask a question to see how the model forms its answer through layers!_")

            with gr.Column(scale=1):
                gr.Markdown("### ℹ️ Quick Guide")
                gr.Markdown("""
                **How to use**:
                1. Ask a one-word answerable question
                2. See how the model forms its answer through all 22 layers
                3. Optionally provide a correction
                4. Compare before/after visualizations

                **Best questions**:
                - What is the capital of [country]?
                - Who is the president of [country]?
                - What color is [object]?
                - When did [event] happen?

                **Example**:
                - Q: "What is the capital of Australia?"
                - A: Watch layers evolve from uncertain → confident
                - Correction: "Actually it's Canberra, not Sydney"
                - See comparison!
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
        # Store original answer to avoid re-generation
        original_answer_cache = {}
        original_viz_cache = {}

        def on_generate(question):
            """Generate answer and show layer-by-layer visualization."""
            if not question.strip():
                return {
                    response_output: "Please enter a question!",
                    turn_summary: "_Ask a question to see how the model forms its answer!_",
                    correction_input: gr.update(visible=False),
                    correction_btn: gr.update(visible=False),
                    comparison_output: gr.update(visible=False)
                }

            # Generate with layer-by-layer tracking
            answer, visualization = generate_one_word_answer(question)

            # Cache for correction comparison
            original_answer_cache[question] = answer
            original_viz_cache[question] = visualization

            # Show correction input after generation
            return {
                response_output: answer,
                turn_summary: visualization,
                correction_input: gr.update(visible=True),
                correction_btn: gr.update(visible=True),
                comparison_output: gr.update(visible=False)
            }

        def on_correction(question, correction):
            """Handle correction and show comparison."""
            if not correction.strip():
                return gr.update(value="Please enter a correction!", visible=True)

            # Get original answer from cache (don't re-generate!)
            original_answer = original_answer_cache.get(question, "Unknown")
            original_viz = original_viz_cache.get(question, "_Original visualization not found_")

            # Reformulate the correction as a contextualized question
            # If correction looks like just a word, assume it's the corrected answer
            correction_words = correction.strip().split()

            if len(correction_words) <= 2:
                # Correction is likely just the answer, reformulate as a question
                # Example: "Green" → "What color is green banana?"
                if "what color" in question.lower():
                    corrected_question = question.replace("?", f" when it is {correction}?")
                elif "what is" in question.lower() or "who is" in question.lower():
                    corrected_question = f"{question.rstrip('?')} (corrected: {correction})?"
                else:
                    corrected_question = f"{question.rstrip('?')} Assume the answer is {correction}. What is it?"
            else:
                # Correction is a full sentence/question, use it directly
                corrected_question = correction

            # Generate answer with correction context
            corrected_answer, corrected_viz = generate_one_word_answer(corrected_question)

            # Create comparison
            comparison = f"""
## 📊 Comparison: Original vs. With Correction Context

### Original Question
**Q**: {question}
**Model's Answer**: **{original_answer}**

### With Correction Context
**Corrected Context**: {corrected_question}
**Model's Answer**: **{corrected_answer}**

---

## 🔬 Layer-by-Layer: Original Question

{original_viz}

---

## 🔬 Layer-by-Layer: With Correction Context

{corrected_viz}

---

## 💡 Interpretation

The comparison shows how the model's layer-by-layer predictions differ when:
- **Original**: Answering "{question}"
- **With Context**: Answering with correction context "{correction}"

Watch how different layers activate differently based on the context!
            """

            return gr.update(value=comparison, visible=True)

        generate_btn.click(
            fn=on_generate,
            inputs=[question_input],
            outputs=[response_output, turn_summary, correction_input, correction_btn, comparison_output]
        )

        correction_btn.click(
            fn=on_correction,
            inputs=[question_input, correction_input],
            outputs=[comparison_output]
        )

        def on_reset():
            """Reset session and UI."""
            result = reset_session()
            return (
                result["response"],  # response_output
                result["summary"],   # turn_summary
                gr.update(visible=False, value=""),  # correction_input
                gr.update(visible=False),  # correction_btn
                gr.update(visible=False, value="")  # comparison_output
            )

        reset_btn.click(
            fn=on_reset,
            inputs=[],
            outputs=[response_output, turn_summary, correction_input, correction_btn, comparison_output]
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
