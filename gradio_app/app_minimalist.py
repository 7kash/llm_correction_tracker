"""
LLM Under the Hood - Minimalistic Elegant Interface

Inspired by luxury brand design principles:
- Maximum whitespace
- Minimal color palette
- Clean typography
- Subtle interactions
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
    """Get vocabulary as a list indexed by token ID."""
    vocab_dict = MODEL.tokenizer.get_vocab()
    vocab_size = len(vocab_dict)
    vocab = [""] * vocab_size
    for token_str, token_id in vocab_dict.items():
        vocab[token_id] = token_str
    return vocab


def generate_one_word_answer(question: str, context: str = None) -> Tuple[str, str]:
    """Generate one-word answer with layer-by-layer predictions."""
    model = load_model()
    result = model.generate_one_word_with_layers(question, max_new_tokens=10, context=context)

    vocab = get_vocab_list()
    visualization = create_layer_by_layer_visualization(result, vocab)

    return result["response"], visualization


def create_attention_chart(internals: dict) -> str:
    """Create minimalistic HTML/CSS bar chart for attention."""
    question = internals.get("question", "")
    context = internals.get("context", "")
    attention_percentages = internals.get("attention_percentages", None)

    if not attention_percentages:
        return "<p style='color: #9CA3AF;'>No attention data available</p>"

    # Build word list from question and context
    allowed_words = set()
    if question:
        words = question.lower().replace("?", " ").replace(".", " ").split()
        allowed_words.update(w.strip() for w in words if len(w.strip()) > 1)
    if context:
        words = context.lower().replace(":", " ").replace(".", " ").replace('"', " ").split()
        allowed_words.update(w.strip() for w in words if len(w.strip()) > 1)

    # Get tokens and filter
    input_tokens = internals.get("input_tokens", [])
    from visualizations.answer_flow import get_clean_words
    all_words = get_clean_words(input_tokens)

    # Calculate attention per word
    word_attention = {}
    for word, token_indices in all_words:
        word_lower = word.lower().strip()
        if word_lower in allowed_words:
            total_attn = sum(attention_percentages[idx] for idx in token_indices if idx < len(attention_percentages))
            if total_attn > 0:
                word_attention[word] = total_attn

    if not word_attention:
        return "<p style='color: #9CA3AF;'>No significant attention detected</p>"

    # Sort and normalize
    sorted_words = sorted(word_attention.items(), key=lambda x: x[1], reverse=True)[:8]
    max_attn = sorted_words[0][1]

    # Create minimalistic chart
    html = """
    <div style="margin: 2rem 0;">
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
    """

    for word, attn in sorted_words:
        pct = int((attn / max_attn) * 100)
        html += f"""
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="min-width: 120px; text-align: right; font-size: 0.875rem; color: #374151; font-weight: 500;">{word}</span>
            <div style="flex: 1; background: #F3F4F6; height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="width: {pct}%; height: 100%; background: #111827; transition: width 0.3s ease;"></div>
            </div>
            <span style="min-width: 45px; font-size: 0.75rem; color: #9CA3AF;">{pct}%</span>
        </div>
        """

    html += """
        </div>
    </div>
    """

    return html


def create_softmax_chart(internals: dict) -> str:
    """Create minimalistic HTML/CSS chart for softmax."""
    softmax_example = internals.get("softmax_example", None)

    if not softmax_example:
        return "<p style='color: #9CA3AF;'>No softmax data available</p>"

    # Create minimalistic chart
    html = """
    <div style="margin: 2rem 0;">
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
    """

    max_prob = max(item["probability"] for item in softmax_example)

    for item in softmax_example:
        token = item["token"]
        logit = item["logit"]
        prob = item["probability"]
        pct = int((prob / max_prob) * 100)

        html += f"""
        <div style="display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0;">
            <span style="min-width: 100px; text-align: right; font-size: 0.875rem; color: #374151; font-weight: 500;">{token}</span>
            <div style="flex: 1; display: flex; flex-direction: column; gap: 0.25rem;">
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <span style="min-width: 80px; font-size: 0.75rem; color: #9CA3AF;">logit: {logit:+.2f}</span>
                    <div style="flex: 1; background: #F3F4F6; height: 4px; border-radius: 2px; overflow: hidden;">
                        <div style="width: {pct}%; height: 100%; background: #111827; transition: width 0.3s ease;"></div>
                    </div>
                    <span style="min-width: 60px; font-size: 0.75rem; color: #374151; font-weight: 500;">{prob*100:.1f}%</span>
                </div>
            </div>
        </div>
        """

    html += """
        </div>
    </div>
    """

    return html


def main_interface():
    # Minimalistic theme - monochrome with subtle accents
    minimal_theme = gr.themes.Monochrome(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        font=("SF Pro Display", "system-ui", "sans-serif"),
    ).set(
        body_background_fill="white",
        body_background_fill_dark="#0A0A0A",
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
        /* Minimalistic, elegant styling */
        body {
            font-family: 'SF Pro Display', system-ui, sans-serif;
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
        }

        .title-section {
            text-align: center;
            padding: 4rem 2rem 3rem 2rem;
            margin-bottom: 3rem;
        }

        .title-main {
            font-size: 2rem;
            font-weight: 300;
            letter-spacing: -0.02em;
            color: #111827;
            margin: 0 0 1rem 0;
        }

        .title-subtitle {
            font-size: 1rem;
            font-weight: 400;
            color: #6B7280;
            margin: 0;
            line-height: 1.6;
        }

        .section-header {
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9CA3AF;
            margin: 3rem 0 1rem 0;
        }

        .theory-box {
            background: #FAFAFA;
            border-left: 2px solid #111827;
            padding: 1.5rem;
            margin: 2rem 0;
        }

        .theory-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: #111827;
            margin: 0 0 0.75rem 0;
        }

        .theory-content {
            font-size: 0.875rem;
            line-height: 1.8;
            color: #4B5563;
        }

        .guide-inline {
            background: #FAFAFA;
            padding: 1.5rem;
            margin: 2rem 0;
            font-size: 0.875rem;
            color: #6B7280;
            line-height: 1.8;
        }

        /* Smaller, minimal buttons */
        .btn-minimal {
            padding: 0.5rem 1.5rem !important;
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            border-radius: 2px !important;
        }

        /* Clean input fields */
        input, textarea {
            border: 1px solid #E5E7EB !important;
            border-radius: 2px !important;
            font-size: 0.875rem !important;
        }

        /* Generous whitespace */
        .gr-padded {
            padding: 2rem 0;
        }

        /* Math formulas */
        .formula {
            font-family: 'Georgia', serif;
            font-style: italic;
            font-size: 0.875rem;
            color: #374151;
            margin: 1rem 0;
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

        # Embedded guide (not separate column)
        gr.HTML("""
        <div class="guide-inline">
            <strong>How it works:</strong> Ask a question with a one-word answer. Watch the model's attention patterns,
            layer-by-layer predictions, and probability distributions. Correct the model to see how it adapts its internal representations.
            <br><br>
            <strong>Example:</strong> "What color is banana?" → Model answers → Tell it "That's wrong" or provide the correct answer → Compare before and after.
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Query"):
                gr.HTML('<p class="section-header">Ask a Question</p>')

                question_input = gr.Textbox(
                    label="",
                    placeholder="What is the capital of Australia?",
                    lines=1,
                    show_label=False
                )

                generate_btn = gr.Button("Generate", size="sm", elem_classes=["btn-minimal"])

                response_output = gr.Textbox(
                    label="Model's Answer",
                    lines=1,
                    interactive=False
                )

                turn_summary = gr.Markdown("")

            with gr.Tab("Correction"):
                gr.HTML('<p class="section-header">Provide Correction</p>')

                gr.Markdown("_If the model's answer was incorrect, you can provide feedback._")

                wrong_btn = gr.Button("That's Wrong", size="sm", elem_classes=["btn-minimal"])

                gr.Markdown("_Or provide the correct answer:_")

                correction_input = gr.Textbox(
                    label="",
                    placeholder="Correct answer",
                    lines=1,
                    show_label=False
                )

                correction_btn = gr.Button("Submit Correction", size="sm", elem_classes=["btn-minimal"])

                comparison_output = gr.Markdown("")

            with gr.Tab("Theory"):
                gr.HTML("""
                <div class="theory-box">
                    <h3 class="theory-title">What happens when you correct an LLM?</h3>
                    <div class="theory-content">
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
                    <h3 class="theory-title">Attention Mechanism</h3>
                    <div class="theory-content">
                        <p>The model assigns weights to each input token to determine relevance.</p>
                        <div class="formula">
                            Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V
                        </div>
                        <p>Higher attention scores indicate stronger influence on the output.</p>
                    </div>
                </div>

                <div class="theory-box">
                    <h3 class="theory-title">Softmax Transformation</h3>
                    <div class="theory-content">
                        <p>Raw model outputs (logits) are transformed into probabilities:</p>
                        <div class="formula">
                            softmax(z<sub>i</sub>) = exp(z<sub>i</sub>) / Σ exp(z<sub>j</sub>)
                        </div>
                        <p>This ensures outputs sum to 1.0 and amplifies differences between scores.</p>
                    </div>
                </div>

                <div class="theory-box">
                    <h3 class="theory-title">Logit Lens</h3>
                    <div class="theory-content">
                        <p>The logit lens technique reveals intermediate predictions by projecting hidden states from any layer through the final prediction head.</p>
                        <p>Early layers show uncertainty; later layers converge on the final answer as the model refines its representation.</p>
                    </div>
                </div>
                """)

        # Event handlers (simplified)
        original_answer_cache = {}
        original_viz_cache = {}

        def on_generate(question):
            answer, visualization = generate_one_word_answer(question)
            original_answer_cache[question] = answer
            original_viz_cache[question] = visualization
            return {
                response_output: answer,
                turn_summary: visualization
            }

        def on_correction(question, correction):
            original_answer = original_answer_cache.get(question, "")
            original_viz = original_viz_cache.get(question, "")

            context = f"That's wrong. The correct answer is: {correction}."
            corrected_answer, corrected_viz = generate_one_word_answer(question, context=context)

            comparison = f"""
## Comparison: Before and After Correction

**Original Answer:** {original_answer}
**Corrected Answer:** {corrected_answer}

---

{corrected_viz}
            """

            return comparison

        generate_btn.click(
            fn=on_generate,
            inputs=[question_input],
            outputs=[response_output, turn_summary]
        )

        correction_btn.click(
            fn=on_correction,
            inputs=[question_input, correction_input],
            outputs=[comparison_output]
        )

        return demo


if __name__ == "__main__":
    demo = main_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
