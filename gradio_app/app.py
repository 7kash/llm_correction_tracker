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


def generate_one_word_answer(question: str, context: str = None) -> Tuple[str, str]:
    """
    Generate one-word answer and show layer-by-layer predictions (logit lens).

    Parameters
    ----------
    question : str
        The question to answer
    context : str, optional
        Additional context to prepend (e.g., "The answer is Green")

    Returns
    -------
    answer : str
    visualization : str (markdown)
    """
    # Validate question (skip validation if context is provided, as it might change the format)
    if not context:
        is_valid, message = is_one_word_question(question)

        if not is_valid:
            error_viz = f"## ⚠️ Question Not Suitable\n\n{message}\n\n**Examples of good questions:**\n- What is the capital of Australia?\n- Who is the president of USA?\n- What color is the sky?\n"
            return "❌ Please ask a one-word answerable question", error_viz
    else:
        message = ""

    model = load_model()

    # Generate with layer-by-layer tracking
    result = model.generate_one_word_with_layers(question, max_new_tokens=10, context=context)

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
    # Minimalistic monochrome theme
    minimal_theme = gr.themes.Monochrome(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        font=("system-ui", "sans-serif"),
    ).set(
        body_background_fill="white",
        button_primary_background_fill="#111827",
        button_primary_background_fill_hover="#374151",
        button_primary_text_color="white",
        button_secondary_background_fill="white",
        button_secondary_background_fill_hover="#F9FAFB",
        button_secondary_border_color="#E5E7EB",
        button_secondary_text_color="#111827",
        input_background_fill="#FAFAFA",
        block_border_width="0px",
        block_shadow="none",
    )

    with gr.Blocks(
        title="LLM under the hood",
        theme=minimal_theme,
        css="""
        /* Minimalistic elegant styling */
        .container {
            max-width: 960px;
            margin: 0 auto;
        }
        .title-section {
            text-align: center;
            padding: 4rem 2rem 2rem 2rem;
            margin-bottom: 2rem;
        }
        .title-main {
            font-size: 1.875rem;
            font-weight: 300;
            letter-spacing: -0.02em;
            color: #111827;
            margin: 0 0 0.75rem 0;
        }
        .title-subtitle {
            font-size: 0.9375rem;
            font-weight: 400;
            color: #6B7280;
            margin: 0;
            line-height: 1.6;
            max-width: 600px;
            margin: 0 auto;
        }
        .guide-inline {
            background: #FAFAFA;
            padding: 1.5rem;
            margin: 2rem auto;
            font-size: 0.875rem;
            color: #6B7280;
            line-height: 1.8;
            max-width: 800px;
            border-left: 2px solid #111827;
        }
        .section-divider {
            height: 1px;
            background: #E5E7EB;
            margin: 3rem 0;
        }
        /* Smaller buttons */
        button {
            padding: 0.5rem 1.25rem !important;
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            border-radius: 2px !important;
        }
        /* Clean inputs */
        input, textarea {
            border: 1px solid #E5E7EB !important;
            border-radius: 2px !important;
            font-size: 0.875rem !important;
        }
        /* Theory boxes */
        .theory-box {
            background: #FAFAFA;
            border-left: 2px solid #111827;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        """
    ) as demo:

        gr.HTML("""
        <div class="title-section">
            <h1 class="title-main">LLM under the hood</h1>
            <p class="title-subtitle">
                Observe the internal mechanisms of language models as they process information.<br>
                See what happens when you tell an LLM it's wrong.
            </p>
        </div>
        """)

        gr.HTML("""
        <div class="guide-inline">
            <strong>How it works:</strong> Ask a question with a one-word answer. Watch the model's attention patterns,
            layer-by-layer predictions, and probability distributions. Correct the model to see how it adapts its internal representations.
            <br><br>
            <strong>Example:</strong> "What color is banana?" → Model answers → Tell it "That's wrong" or provide the correct answer → Compare before and after.
            <br><br>
            <strong>Model:</strong> TinyLlama-1.1B (22 layers, ~2-5 seconds per response)
        </div>
        """)

        gr.HTML('<div class="section-divider"></div>')

        with gr.Tabs():
            with gr.Tab("Query"):
                question_input = gr.Textbox(
                    label="",
                    placeholder="What is the capital of Australia?",
                    lines=1,
                    show_label=False,
                    container=False
                )

                generate_btn = gr.Button("Generate", variant="primary", size="sm")

                response_output = gr.Textbox(
                    label="Model's Answer",
                    lines=1,
                    interactive=False
                )

                turn_summary = gr.Markdown("")

            with gr.Tab("Correction"):
                gr.Markdown("_If the model's answer was incorrect, provide feedback:_")

                with gr.Row():
                    wrong_btn = gr.Button("That's Wrong", variant="stop", size="sm", scale=1)
                    correction_input = gr.Textbox(
                        label="",
                        placeholder="Or type the correct answer",
                        lines=1,
                        show_label=False,
                        scale=2
                    )

                correction_btn = gr.Button("Submit Correction", variant="secondary", size="sm")

                comparison_output = gr.Markdown("")

            with gr.Tab("Theory"):
                gr.HTML("""
                <div class="theory-box">
                    <h3 style="font-size: 0.875rem; font-weight: 500; color: #111827; margin: 0 0 1rem 0;">
                        What happens when you correct an LLM?
                    </h3>
                    <div style="font-size: 0.875rem; line-height: 1.8; color: #4B5563;">
                        <p>When you provide a correction, the model processes it as additional context. This triggers several internal changes:</p>
                        <ol>
                            <li><strong>Attention Reweighting:</strong> The model shifts attention to correction-related tokens</li>
                            <li><strong>Feature Activation:</strong> Error-detection and disagreement neurons activate</li>
                            <li><strong>Layer Progression:</strong> Early layers process the contradiction, later layers synthesize the correction</li>
                        </ol>
                        <p>The visualizations show these changes quantitatively through attention distributions and layer-wise predictions.</p>
                    </div>
                </div>

                <div class="theory-box">
                    <h3 style="font-size: 0.875rem; font-weight: 500; color: #111827; margin: 0 0 1rem 0;">
                        Attention Mechanism
                    </h3>
                    <div style="font-size: 0.875rem; line-height: 1.8; color: #4B5563;">
                        <p>The model assigns weights to each input token to determine relevance.</p>
                        <p style="font-family: Georgia, serif; font-style: italic; margin: 1rem 0;">
                            Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V
                        </p>
                        <p>Higher attention scores indicate stronger influence on the output.</p>
                    </div>
                </div>

                <div class="theory-box">
                    <h3 style="font-size: 0.875rem; font-weight: 500; color: #111827; margin: 0 0 1rem 0;">
                        Softmax Transformation
                    </h3>
                    <div style="font-size: 0.875rem; line-height: 1.8; color: #4B5563;">
                        <p>Raw model outputs (logits) are transformed into probabilities:</p>
                        <p style="font-family: Georgia, serif; font-style: italic; margin: 1rem 0;">
                            softmax(z<sub>i</sub>) = exp(z<sub>i</sub>) / Σ exp(z<sub>j</sub>)
                        </p>
                        <p>This ensures outputs sum to 1.0 and amplifies differences between scores.</p>
                    </div>
                </div>

                <div class="theory-box">
                    <h3 style="font-size: 0.875rem; font-weight: 500; color: #111827; margin: 0 0 1rem 0;">
                        Logit Lens
                    </h3>
                    <div style="font-size: 0.875rem; line-height: 1.8; color: #4B5563;">
                        <p>The logit lens technique reveals intermediate predictions by projecting hidden states from any layer through the final prediction head.</p>
                        <p>Early layers show uncertainty; later layers converge on the final answer as the model refines its representation.</p>
                    </div>
                </div>
                """)


        # Event handlers
        # Store original answer to avoid re-generation
        original_answer_cache = {}
        original_viz_cache = {}

        def create_comparison_view(question, original_answer, original_viz, correction_context, corrected_answer, corrected_viz):
            """Create minimalistic comparison view - show answer comparison first, then full corrected visualization."""

            # Show answer comparison first (what user wants to see)
            html = f"""
## Answer Comparison

<div style="background: #FAFAFA; padding: 1.5rem; margin: 2rem 0; border-left: 2px solid #111827;">
    <div style="display: flex; gap: 2rem; margin-bottom: 1rem;">
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #9CA3AF; margin-bottom: 0.5rem;">
                Original Answer
            </div>
            <div style="font-size: 1.25rem; font-weight: 500; color: #111827;">
                {original_answer}
            </div>
        </div>
        <div style="flex: 1;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #9CA3AF; margin-bottom: 0.5rem;">
                After Correction
            </div>
            <div style="font-size: 1.25rem; font-weight: 500; color: #111827;">
                {corrected_answer}
            </div>
        </div>
    </div>
    <div style="font-size: 0.875rem; color: #6B7280; margin-top: 1rem;">
        Correction context: "{correction_context}"
    </div>
</div>

---

## Corrected Response Analysis

{corrected_viz}
            """

            return html

        def on_generate(question):
            """Generate answer and show layer-by-layer visualization."""
            if not question.strip():
                return ("Please enter a question!", "_Ask a question to see how the model forms its answer!_")

            # Generate with layer-by-layer tracking
            answer, visualization = generate_one_word_answer(question)

            # Cache for correction comparison
            original_answer_cache[question] = answer
            original_viz_cache[question] = visualization

            return (answer, visualization)

        def on_wrong_clicked(question):
            """Handle 'That's Wrong!' button - tell model its answer was wrong."""
            if not question or question not in original_answer_cache:
                return "⚠️ Please generate an answer first!"

            # Get original answer from cache
            original_answer = original_answer_cache[question]
            original_viz = original_viz_cache[question]

            # Use "That's wrong" without providing correct answer
            correction_context = "That's wrong. Try again."

            # Generate answer with correction context
            corrected_answer, corrected_viz = generate_one_word_answer(question, context=correction_context)

            # Create comparison
            comparison = create_comparison_view(
                question, original_answer, original_viz,
                correction_context, corrected_answer, corrected_viz
            )

            return comparison

        def on_correction(question, correction):
            """Handle manual correction with specific answer."""
            if not correction.strip():
                return "⚠️ Please enter a correction!"

            if not question or question not in original_answer_cache:
                return "⚠️ Please generate an answer first!"

            # Get original answer from cache
            original_answer = original_answer_cache[question]
            original_viz = original_viz_cache[question]

            # Build context
            if len(correction.split()) <= 2:
                correction_context = f"That's wrong. The correct answer is: {correction}."
            else:
                correction_context = f"That's wrong. {correction}"

            # Generate answer with correction context
            corrected_answer, corrected_viz = generate_one_word_answer(question, context=correction_context)

            # Create comparison
            comparison = create_comparison_view(
                question, original_answer, original_viz,
                correction_context, corrected_answer, corrected_viz
            )

            return comparison

        generate_btn.click(
            fn=on_generate,
            inputs=[question_input],
            outputs=[response_output, turn_summary],
            show_progress="full"
        )

        wrong_btn.click(
            fn=on_wrong_clicked,
            inputs=[question_input],
            outputs=[comparison_output],
            show_progress="full"
        )

        correction_btn.click(
            fn=on_correction,
            inputs=[question_input, correction_input],
            outputs=[comparison_output],
            show_progress="full"
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
