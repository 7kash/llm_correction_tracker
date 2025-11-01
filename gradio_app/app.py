"""
LLM Inference Tracker - Gradio App

Track how LLMs change their responses when corrected, with visualizations
of internal mechanisms: attention, layer trajectories, and logit lens.

Deployment: Hugging Face Spaces
"""

import gradio as gr
import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import html

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
    analyze_word_importance,
    get_clean_words,
    clean_token
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


def _format_percentage_bar(label: str, value: float, color_var: str, max_items: int = 8) -> str:
    """Create a horizontal bar for percentage-based metrics."""
    display_pct = round(value, 1)
    bar_width = min(100, int(value))
    return (
        "<div class=\"metric-row\">"
        f"<span class=\"metric-label\">{label}</span>"
        "<div class=\"metric-track\">"
        f"<div class=\"metric-fill\" style=\"width: {bar_width}%; background: var({color_var});\"></div>"
        "</div>"
        f"<span class=\"metric-value\">{display_pct}%</span>"
        "</div>"
    )


def _collect_attention_metrics(internals: dict) -> Tuple[List[Tuple[str, float]], str]:
    """Return top attention words and formatted HTML."""
    attention_percentages = internals.get("attention_percentages")
    if attention_percentages is None:
        return [], "_Attention data is not available for this turn yet._"

    input_tokens = internals.get("input_tokens", [])
    question = internals.get("question", "")
    context = internals.get("context", "")

    allowed_words = set()
    if question:
        for word in question.lower().replace("?", " ").replace(",", " ").split():
            word = word.strip()
            if word:
                allowed_words.add(word)

    if context:
        context_clean = (
            context.replace("#", " ")
            .replace(",", " ")
            .replace("?", " ")
            .replace(".", " ")
            .replace("\"", " ")
            .replace("'", " ")
        )
        for word in context_clean.split():
            word = word.strip().lower()
            if word:
                allowed_words.add(word)

    word_attention: Dict[str, float] = {}
    for word, token_indices in get_clean_words(input_tokens):
        word_lower = word.lower().strip()
        if not word_lower or word_lower not in allowed_words:
            continue

        total_attention = 0.0
        for idx in token_indices:
            if idx < len(attention_percentages):
                total_attention += float(attention_percentages[idx])

        if total_attention > 0:
            word_attention[word] = total_attention

    if not word_attention:
        return [], "_The model focused on helper tokens rather than the question words. Try another prompt!_"

    sorted_words = sorted(word_attention.items(), key=lambda item: item[1], reverse=True)[:8]
    metric_rows = [
        _format_percentage_bar(word, value, "--attention-color")
        for word, value in sorted_words
    ]
    return sorted_words, "<div class=\"metric-list\">" + "".join(metric_rows) + "</div>"


def _collect_softmax_metrics(internals: dict) -> Tuple[List[dict], str]:
    """Return softmax token candidates and formatted HTML."""
    softmax_example = internals.get("softmax_example")
    if not softmax_example:
        return [], "_Probability data will appear once the model proposes a token._"

    items = []
    rows = []
    for item in softmax_example:
        token = item.get("token", "?")
        prob = float(item.get("probability", 0.0)) * 100
        logit = float(item.get("logit", 0.0))
        cleaned = clean_token(token)
        if cleaned:
            token_display, _ = cleaned
        else:
            token_display = token.replace("▁", " ").strip() or token

        items.append({
            "token": token_display,
            "prob": prob,
            "logit": logit,
        })
        bar_width = min(100, int(prob))
        rows.append(
            "<div class=\"metric-row\">"
            f"<span class=\"metric-label\">{token_display}</span>"
            f"<span class=\"metric-logit\">logit {logit:+.2f}</span>"
            "<div class=\"metric-track\">"
            f"<div class=\"metric-fill\" style=\"width: {bar_width}%; background: var(--softmax-color);\"></div>"
            "</div>"
            f"<span class=\"metric-value\">{prob:.1f}%</span>"
            "</div>"
        )

    return items, "<div class=\"metric-list\">" + "".join(rows) + "</div>"


def _collect_layer_metrics(internals: dict, actual_answer: str) -> Tuple[List[dict], str]:
    """Return layer-by-layer predictions and formatted HTML."""
    layer_predictions = internals.get("layer_predictions") or []
    if not layer_predictions:
        return [], "_Layer probes activate once the model begins forming an answer._"

    layer_rows = []
    metrics = []
    for layer_idx, layer_data in enumerate(layer_predictions):
        predictions = layer_data.get("predictions", [])
        actual_entry = None
        alternatives = []

        for pred in predictions:
            token = pred.get("token", "")
            prob = float(pred.get("probability", 0.0)) * 100
            is_actual = pred.get("is_actual_answer", False)
            cleaned = clean_token(token)
            if cleaned:
                display_token, _ = cleaned
            else:
                display_token = token.replace("▁", " ").strip() or token

            data_point = {"token": display_token, "prob": prob}
            if is_actual:
                actual_entry = data_point
            else:
                alternatives.append(data_point)

        alternatives = alternatives[:3]
        metrics.append({
            "layer": layer_idx,
            "actual": actual_entry,
            "alternatives": alternatives,
        })

        alt_lines = "".join(
            f"<li>{alt['token']} — {alt['prob']:.1f}%</li>" for alt in alternatives
        ) or "<li>No strong alternatives</li>"

        actual_line = (
            f"<strong>{actual_entry['token']}</strong> — {actual_entry['prob']:.1f}%"
            if actual_entry
            else "Model had not locked onto the final word yet"
        )

        layer_rows.append(
            "<div class=\"layer-card\">"
            f"<div class=\"layer-title\">Layer {layer_idx}</div>"
            f"<div class=\"layer-actual\">{actual_line}</div>"
            f"<ul class=\"layer-alternatives\">{alt_lines}</ul>"
            "</div>"
        )

    return metrics, "<div class=\"layer-grid\">" + "".join(layer_rows) + "</div>"


THEORY_TEXT = {
    "overview": """
### How a transformer answers your question

<div class=\"section-note pill-softmax\"><strong>High-level flow:</strong> during pre-training TinyLlama learned statistical patterns from billions of words. Those frozen weights now transform each new prompt into an internal state that points toward the next word.</div>

1. **Embedding & position sense.** The text is split into sub-word pieces and turned into vectors that encode meaning plus order.
2. **Attention exchanges.** Learned projection matrices create queries, keys, and values for every token so each layer can borrow context from the rest of the prompt.
3. **Layer updates.** Feed-forward blocks transform the focused information into sharper internal features before handing them to the next attention stage.
4. **Prediction head.** The refined vector is scored against every vocabulary item with the language-model head, then softmax converts those scores into a true probability distribution.

This app surfaces the three most important snapshots: attention weights (what the multi-head attention layers focused on), the softmax peaks (which words competed), and the layer-by-layer probes (how certainty built up as the signal flowed through the trained weights).

During training, gradient descent nudged billions of weights so that these steps reproduce patterns found in text. At inference time we simply reuse those weights, letting linear algebra and the learned attention heads carry the question toward its answer.
""".strip(),
    "feedback": """
### What happens when you correct the model

<div class=\"section-note pill-layers\"><strong>Why feedback matters:</strong> telling the model it is wrong does not change its weights, but it augments the next prompt. The conversation history now includes your reminder (and any corrected word you typed), so attention, layer activations, and softmax scores shift toward the correction.</div>

* **Attention.** The reminder phrase pulls focus toward the corrective words—or at least flags that the previous answer was wrong—so the model rereads the prompt with that constraint in mind.
* **Layer dynamics.** Early layers incorporate the new clue, while deeper layers propagate it, pushing the correct answer higher.
* **Softmax.** Competing tokens lose probability mass as the corrected token gathers stronger evidence.

Even though the weights stay fixed, the extra text behaves like temporary memory: every new turn recomputes the internals using both the original question and your feedback, effectively giving the network more context to reason with.
That is why the comparison tab shows side-by-side internals—the only difference is the added context you supplied in natural language.
""".strip(),
    "attention": """
### How attention works

<div class=\"section-note pill-attention\"><strong>Plain-language intuition:</strong> attention assigns a weight to each input word so the model can concentrate on the parts of the question that matter most for the next word it will say.</div>

**Formula:**

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

Step-by-step:

1. Learned matrices turn the current hidden state into a **query (Q)** and every prompt token into matching **keys (K)** and **values (V)**.
2. Compare the query with each key to see which tokens feel related, then scale the scores by $1/\\sqrt{d_k}$ so large vectors do not explode the numbers.
3. Use softmax to turn those scores into weights that sum to 1.
4. Blend the value vectors using those weights, giving the model a context summary tailored to the next word.

The attention bars in the Query tab show which words grabbed the most weight for the final answer.
""".strip(),
    "softmax": """
### How softmax turns scores into probabilities

<div class=\"section-note pill-softmax\"><strong>Plain-language intuition:</strong> the model assigns a raw score to every possible next token. Softmax stretches those scores so that they add up to 100%, making it easy to see which word is most likely.</div>

**Formula:**

$$\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

What this means in practice:

* Each candidate token gets exponentiated ($e^{z_i}$), exaggerating stronger evidence.
* All exponentiated scores are normalised by their sum so they form a true probability distribution.
* Even small score gaps can create big probability differences, which is why the probability bars can look decisive.

For this demo we disable sampling and set a low temperature (0.1) so the probabilities you see come directly from the model's frozen weights rather than randomness.
""".strip(),
    "layers": """
### Why layer-by-layer probes are helpful

<div class=\"section-note pill-layers\"><strong>Plain-language intuition:</strong> each transformer layer refines the hidden representation of the answer. Probing with the logit lens—temporarily attaching the final prediction head to every layer—shows how confidence in the final word grows as we move deeper into the network.</div>

TinyLlama has 22 transformer layers. For every layer we temporarily plug in the final prediction head and read the implied probabilities:

* **Early layers** capture broad topics (“this is about geography”). The correct word may only have a small lead.
* **Middle layers** integrate attention results and rule out conflicting options.
* **Late layers** sharpen the concept until one token clearly dominates.

When feedback is applied, look for the layer cards to show the corrected token rising sooner or more steeply across the stack.
""".strip(),
}


def build_visualization_payload(internals: dict, guidance_note: str = "") -> dict:
    """Bundle friendly explanations, metrics, and theory snippets for the UI."""

    question = internals.get("question", "").strip() or "your question"
    answer = internals.get("response", "").strip() or internals.get("text", "")

    overview_lines = [
        "### What the model just did",
        (
            "TinyLlama was trained before it ever met you: its frozen weights already store facts and language patterns. "
            "When you now ask **{question}**, the model replays that training to predict the next word."
        ).format(question=question),
        (
            "It combined your question with any earlier turns, pushed the tokens through 22 attention-plus-feed-forward layers, "
            "and finally proposed **{answer or '...'}** as the most likely word."
        ).format(answer=answer),
        "The panels below show how the prompt shaped attention, how the softmax head weighed competing words, and how certainty grew across the layers.",
    ]
    if guidance_note:
        overview_lines.append(f"<div class='soft-note'>{guidance_note}</div>")

    attention_metrics, attention_html = _collect_attention_metrics(internals)
    softmax_metrics, softmax_html = _collect_softmax_metrics(internals)
    layer_metrics, layer_html = _collect_layer_metrics(internals, answer)

    return {
        "question": question,
        "answer": answer,
        "overview": "\n\n".join(overview_lines),
        "attention": {
            "metrics": attention_metrics,
            "markdown": (
                "<div class=\"section-note pill-attention\"><strong>Where focus went:</strong> higher bars mean the model relied more on that word.</div>"
                + attention_html
            ),
        },
        "softmax": {
            "metrics": softmax_metrics,
            "markdown": (
                "<div class=\"section-note pill-softmax\"><strong>Top candidates:</strong> the probability shows how confident the model was before choosing.</div>"
                + softmax_html
            ),
        },
        "layers": {
            "metrics": layer_metrics,
            "markdown": (
                "<div class=\"section-note pill-layers\"><strong>Confidence by layer:</strong> later layers should favour the final answer.</div>"
                + layer_html
            ),
        },
        "theory": {
            "overview": THEORY_TEXT["overview"],
            "feedback": THEORY_TEXT["feedback"],
            "attention": THEORY_TEXT["attention"],
            "softmax": THEORY_TEXT["softmax"],
            "layers": THEORY_TEXT["layers"],
        },
    }


def build_correction_sections(
    question: str,
    original_answer: str,
    original_payload: dict,
    correction_context: str,
    corrected_answer: str,
    corrected_payload: dict,
) -> Tuple[str, str, str, str]:
    """Return summary and per-section comparison HTML blocks."""

    def extract_markdown(payload: dict, key: str) -> str:
        if not isinstance(payload, dict):
            return "<em>No data available.</em>"
        section = payload.get(key, {})
        if not isinstance(section, dict):
            return "<em>No data available.</em>"
        raw_markdown = section.get("markdown", "<em>No data available.</em>")
        if "section-note" in raw_markdown:
            parts = raw_markdown.split("</div>", 1)
            if len(parts) == 2:
                return parts[1]
        return raw_markdown

    safe_question = html.escape(question)
    safe_original = html.escape(original_answer or "—")
    safe_corrected = html.escape(corrected_answer or "—")
    safe_context = html.escape(correction_context or "(no extra hint supplied)")

    combined_prompt = " ".join(
        token.strip()
        for token in [question or "", correction_context or ""]
        if token and token.strip()
    )
    safe_prompt = html.escape(combined_prompt or question)

    summary = f"""
<div class="section-note pill-layers">
    <strong>Question:</strong> {safe_question}<br>
    <strong>Before feedback:</strong> {safe_original}<br>
    <strong>After feedback:</strong> {safe_corrected}
</div>
<p style="font-size:0.9rem;color:var(--muted-text);">Your correction was appended to the next prompt as: <code>{safe_context}</code>. That extra line gives the model more context when it re-reads the question.</p>
"""

    attention_intro = f"""
<div class="section-note pill-attention"><strong>Combined prompt on replay:</strong>
    <div style="margin-top:0.4rem;">
        <div><strong>Question:</strong> {safe_question}</div>
        <div><strong>Added context:</strong> {safe_context}</div>
        <div style="margin-top:0.4rem;"><strong>Sent to TinyLlama:</strong> <code>{safe_prompt}</code></div>
    </div>
    The added line reminds the model which answer to avoid and layers in any corrected wording, so the attention pattern balances the original question with your guidance.
</div>
"""

    attention = attention_intro + f"""
<div class="comparison-card">
    <div class="column">
        <h4 style="margin-top:0; color: var(--attention-color);">Before feedback</h4>
        {extract_markdown(original_payload, 'attention')}
    </div>
    <div class="column">
        <h4 style="margin-top:0; color: var(--attention-color);">After feedback</h4>
        {extract_markdown(corrected_payload, 'attention')}
    </div>
</div>
"""

    softmax_intro = """
<div class="section-note pill-softmax"><strong>How probabilities change:</strong> The left column shows the raw scores from the first run. The right column reruns the frozen weights with your reminder, so probability mass slides toward the corrected word.</div>
"""

    softmax = softmax_intro + f"""
<div class="comparison-card">
    <div class="column">
        <h4 style="margin-top:0; color: var(--softmax-color);">Before feedback</h4>
        {extract_markdown(original_payload, 'softmax')}
    </div>
    <div class="column">
        <h4 style="margin-top:0; color: var(--softmax-color);">After feedback</h4>
        {extract_markdown(corrected_payload, 'softmax')}
    </div>
</div>
"""

    layers_intro = """
<div class="section-note pill-layers"><strong>Layer story:</strong> Each card is the logit lens probe for a layer. After the correction, deeper layers usually lock onto the right word sooner because the added context narrows the possibilities.</div>
"""

    layers = layers_intro + f"""
<div class="comparison-card">
    <div class="column">
        <h4 style="margin-top:0; color: var(--layers-color);">Before feedback</h4>
        {extract_markdown(original_payload, 'layers')}
    </div>
    <div class="column">
        <h4 style="margin-top:0; color: var(--layers-color);">After feedback</h4>
        {extract_markdown(corrected_payload, 'layers')}
    </div>
</div>
"""

    return summary, attention, softmax, layers

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
    payload = build_visualization_payload(result, guidance_note=message)

    return result["response"], payload


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
        /* Minimalistic monochrome styling */
        :root {
            --attention-color: #111827;
            --attention-bg: #F3F4F6;
            --softmax-color: #1F2937;
            --softmax-bg: #F9FAFB;
            --layers-color: #4B5563;
            --layers-bg: #F3F4F6;
            --muted-text: #6B7280;
            --surface-border: #E5E7EB;
            --surface: #FFFFFF;
        }

        body, input, textarea, button {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            color: #111827;
        }

        body {
            font-size: 15px;
            line-height: 1.65;
            background: #F3F4F6;
        }

        p, li {
            line-height: 1.7;
        }

        .gradio-container {
            padding-top: 1.25rem !important;
        }

        .container {
            max-width: 960px;
            margin: 0 auto;
        }

        .title-section {
            text-align: center;
            padding: 1.25rem 1.75rem 1.1rem;
            margin-bottom: 0.9rem;
        }

        .title-main {
            font-size: 1.875rem;
            font-weight: 300;
            letter-spacing: -0.02em;
            color: #111827;
            margin: 0 0 0.75rem 0;
        }

        .title-subtitle {
            font-size: 0.95rem;
            font-weight: 400;
            color: var(--muted-text);
            margin: 0 auto;
            line-height: 1.6;
            max-width: 620px;
        }

        .guide-inline {
            background: #F9FAFB;
            padding: 1.2rem 1.5rem;
            margin: 0.85rem auto 1.1rem;
            font-size: 0.95rem;
            color: #1F2937;
            line-height: 1.7;
            max-width: 840px;
            border-left: 2px solid #111827;
            border-radius: 4px;
        }

        .section-divider {
            height: 1px;
            background: var(--surface-border);
            margin: 2.25rem 0;
        }

        button {
            padding: 0.5rem 1.25rem !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            border-radius: 3px !important;
            letter-spacing: 0.01em;
            color: #111827;
        }

        button.secondary {
            color: #111827 !important;
        }

        input, textarea {
            border: 1px solid var(--surface-border) !important;
            border-radius: 3px !important;
            font-size: 0.95rem !important;
            background: #FFFFFF !important;
        }

        .theory-box {
            background: #F9FAFB;
            border-left: 2px solid #111827;
            padding: 1.4rem;
            margin: 1.5rem 0;
            border-radius: 4px;
        }

        .correction-guide {
            background: #FFFFFF;
            border-radius: 6px;
            border: 1px solid var(--surface-border);
            padding: 1.5rem;
            margin-top: 1.5rem;
        }

        .correction-guide h4 {
            margin: 0 0 0.75rem 0;
            font-size: 1.05rem;
            color: #1F2937;
        }

        .correction-guide ul {
            margin: 0.75rem 0 0 1.25rem;
            color: var(--muted-text);
        }

        .comparison-table thead th {
            background: #F3F4F6;
        }

        .comparison-table td {
            background: #FFFFFF;
        }

        .comparison-table tr:not(:last-child) td {
            border-bottom: 1px solid var(--surface-border);
        }

        .soft-note {
            margin-top: 1rem;
            padding: 0.75rem 1rem;
            background: #F3F4F6;
            border-left: 2px solid #D1D5DB;
            color: #1F2937;
            font-size: 0.9rem;
            border-radius: 4px;
        }

        .section-note {
            padding: 0.85rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            margin-bottom: 1.1rem;
            line-height: 1.6;
            background: #F9FAFB;
            border: 1px solid var(--surface-border);
        }

        .pill-attention {
            background: var(--attention-bg);
            border-left: 2px solid var(--attention-color);
        }

        .pill-softmax {
            background: var(--softmax-bg);
            border-left: 2px solid var(--softmax-color);
        }

        .pill-layers {
            background: var(--layers-bg);
            border-left: 2px solid var(--layers-color);
        }

        .metric-list {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: 140px 90px 1fr 60px;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.9rem;
        }

        .metric-label {
            text-align: right;
            font-weight: 500;
            color: #1F2937;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .metric-logit {
            font-family: "JetBrains Mono", "Menlo", monospace;
            font-size: 0.75rem;
            color: var(--muted-text);
        }

        .metric-track {
            position: relative;
            width: 100%;
            height: 6px;
            border-radius: 999px;
            background: #E5E7EB;
            overflow: hidden;
        }

        .metric-fill {
            position: absolute;
            height: 100%;
            left: 0;
            top: 0;
            border-radius: inherit;
        }

        .metric-value {
            font-size: 0.75rem;
            color: var(--muted-text);
            font-weight: 500;
        }

        .layer-grid {
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }

        .layer-card {
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--surface);
            box-shadow: none;
        }

        .layer-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #1F2937;
            margin-bottom: 0.5rem;
        }

        .layer-actual {
            font-size: 0.9rem;
            color: #1F2937;
            margin-bottom: 0.5rem;
        }

        .layer-alternatives {
            margin: 0;
            padding-left: 1rem;
            color: var(--muted-text);
            font-size: 0.85rem;
        }

        .layer-alternatives li {
            margin-bottom: 0.25rem;
        }

        .comparison-card {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
        }

        .comparison-card .column {
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--surface);
        }

        .tab-nav button {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            color: #1F2937;
        }

        #query-subtabs .tab-nav button:nth-of-type(1)::before,
        #correction-subtabs .tab-nav button:nth-of-type(1)::before,
        #theory-subtabs .tab-nav button:nth-of-type(1)::before {
            content: '';
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            background: var(--attention-bg);
            display: inline-block;
            box-shadow: 0 0 0 2px #FFFFFF;
        }

        #query-subtabs .tab-nav button:nth-of-type(2)::before,
        #correction-subtabs .tab-nav button:nth-of-type(2)::before,
        #theory-subtabs .tab-nav button:nth-of-type(2)::before {
            content: '';
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            background: var(--softmax-bg);
            display: inline-block;
            box-shadow: 0 0 0 2px #FFFFFF;
        }

        #query-subtabs .tab-nav button:nth-of-type(3)::before,
        #correction-subtabs .tab-nav button:nth-of-type(3)::before,
        #theory-subtabs .tab-nav button:nth-of-type(3)::before {
            content: '';
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 999px;
            background: var(--layers-bg);
            display: inline-block;
            box-shadow: 0 0 0 2px #FFFFFF;
        }

        #query-subtabs .tab-nav button:nth-of-type(1)[aria-selected="true"],
        #correction-subtabs .tab-nav button:nth-of-type(1)[aria-selected="true"],
        #theory-subtabs .tab-nav button:nth-of-type(1)[aria-selected="true"] {
            background: var(--attention-bg);
        }

        #query-subtabs .tab-nav button:nth-of-type(2)[aria-selected="true"],
        #correction-subtabs .tab-nav button:nth-of-type(2)[aria-selected="true"],
        #theory-subtabs .tab-nav button:nth-of-type(2)[aria-selected="true"] {
            background: var(--softmax-bg);
        }

        #query-subtabs .tab-nav button:nth-of-type(3)[aria-selected="true"],
        #correction-subtabs .tab-nav button:nth-of-type(3)[aria-selected="true"],
        #theory-subtabs .tab-nav button:nth-of-type(3)[aria-selected="true"] {
            background: var(--layers-bg);
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
            <strong>How it works:</strong> TinyLlama was pre-trained on billions of words, so its weights already encode language patterns.
            Each time you ask a question we combine it with any previous turns, run the frozen network through 22 layers, and read out the most likely next word.
            <br><br>
            <strong>Why correction helps:</strong> Pressing “That’s Wrong” adds a follow-up line that says the last answer was incorrect, and if you type the right word we include that too.
            The model re-reads the original question plus that extra context with the same frozen weights, shifting attention, layer activations, and probabilities toward the corrected answer.
            <br><br>
            <strong>Example:</strong> "What color is banana?" → Model answers → Tell it "That's wrong, the right answer is yellow" → Compare how the internals change.
            <br><br>
            <strong>Model:</strong> TinyLlama-1.1B (22 layers, ~2-5 seconds per response)
        </div>
        """)

        gr.HTML('<div class="section-divider"></div>')

        empty_attention_html = "<em>Run the model to see how attention focuses on your question.</em>"
        empty_softmax_html = "<em>Run the model to see which words had the highest probability.</em>"
        empty_layers_html = "<em>Run the model to see how confidence evolves across layers.</em>"
        correction_placeholder = "_Generate an answer and press “That’s Wrong” to compare before and after._"

        with gr.Tabs():
            with gr.Tab("Query"):
                gr.Markdown("### Ask the model a quick fact")
                gr.Markdown("Keep it simple: questions like **What is the capital of Australia?** work best because the answer is a single word. The panels explain, in plain language, how the frozen TinyLlama weights arrived at that word.")

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

                query_overview_md = gr.Markdown("_Run the model to see how the pre-trained network pieces the answer together._")

                with gr.Tabs(elem_id="query-subtabs"):
                    with gr.Tab("Attention"):
                        gr.Markdown("**Attention focus:** These bars show how much of the model's attention landed on each word from your prompt (plus any earlier corrections). Higher percentages mean that token strongly influenced the chosen answer.")
                        query_attention_panel = gr.HTML(empty_attention_html)

                    with gr.Tab("Softmax"):
                        gr.Markdown("**Probability peaks:** After the layers finish, the model scores every vocabulary item. This view highlights the top candidates and how confident TinyLlama was before selecting the final word.")
                        query_softmax_panel = gr.HTML(empty_softmax_html)

                    with gr.Tab("Layer by Layer"):
                        gr.Markdown("**Layer progression:** Each card probes a transformer layer. Early layers carry a hunch, while later layers sharpen the signal. Watch how certainty grows as the computation moves deeper.")
                        query_layers_panel = gr.HTML(empty_layers_html)

            with gr.Tab("Correction"):
                gr.Markdown("### Teach the model when it slips")
                gr.Markdown("First, tell the model its answer was off. The next run prepends a reminder that the last answer was wrong, plus any corrected word you provide, so the frozen weights re-evaluate the prompt with that extra clue.")

                wrong_btn = gr.Button("That's Wrong", variant="stop", size="sm")

                gr.Markdown("**Optional:** Type the correct word so the model can aim directly at it. We attach that word to the follow-up prompt, which shifts the attention and probabilities you will see below.")
                correction_input = gr.Textbox(
                    label="",
                    placeholder="Type the word it should have said",
                    lines=1,
                    show_label=False
                )

                correction_btn = gr.Button("Submit Correction", variant="secondary", size="sm")

                correction_summary_md = gr.Markdown(correction_placeholder)

                with gr.Tabs(elem_id="correction-subtabs"):
                    with gr.Tab("Attention"):
                        gr.Markdown("**Attention shift:** Left is the first run, right is the re-run with your reminder included. Look for the dark highlights to move toward the words you emphasised.")
                        correction_attention_panel = gr.HTML("<em>No comparison yet.</em>")

                    with gr.Tab("Softmax"):
                        gr.Markdown("**Softmax change:** Compare how the probability bars reallocate mass. The corrected word should gain percentage once the hint is in the prompt.")
                        correction_softmax_panel = gr.HTML("<em>No comparison yet.</em>")

                    with gr.Tab("Layer by Layer"):
                        gr.Markdown("**Layer momentum:** The layer cards show each layer's top token. After a good correction, the right word tends to appear earlier and stay dominant across the stack.")
                        correction_layers_panel = gr.HTML("<em>No comparison yet.</em>")

            with gr.Tab("Theory"):
                gr.Markdown("### Peek behind the math")
                gr.Markdown("These notes unpack what the TinyLlama is doing internally, then dive into the colour-coded pieces you see across the app.")

                theory_overview_md = gr.Markdown(THEORY_TEXT["overview"])
                theory_feedback_md = gr.Markdown(THEORY_TEXT["feedback"])

                with gr.Tabs(elem_id="theory-subtabs"):
                    with gr.Tab("Attention"):
                        theory_attention_md = gr.Markdown(THEORY_TEXT["attention"])

                    with gr.Tab("Softmax"):
                        theory_softmax_md = gr.Markdown(THEORY_TEXT["softmax"])

                    with gr.Tab("Layer by Layer"):
                        theory_layers_md = gr.Markdown(THEORY_TEXT["layers"])

        # Event handlers
        # Store original answer to avoid re-generation
        original_answer_cache: Dict[str, Dict[str, object]] = {}

        def on_generate(question):
            """Generate answer and update all tabs."""
            if not question.strip():
                return (
                    "Please enter a question!",
                    "_Ask something like “What is the capital of Australia?”_",
                    empty_attention_html,
                    empty_softmax_html,
                    empty_layers_html,
                    THEORY_TEXT["overview"],
                    THEORY_TEXT["feedback"],
                    THEORY_TEXT["attention"],
                    THEORY_TEXT["softmax"],
                    THEORY_TEXT["layers"],
                    correction_placeholder,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            answer, payload = generate_one_word_answer(question)

            if isinstance(payload, str):
                # Error or guidance message
                return (
                    answer,
                    payload,
                    empty_attention_html,
                    empty_softmax_html,
                    empty_layers_html,
                    THEORY_TEXT["overview"],
                    THEORY_TEXT["feedback"],
                    THEORY_TEXT["attention"],
                    THEORY_TEXT["softmax"],
                    THEORY_TEXT["layers"],
                    correction_placeholder,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            original_answer_cache[question] = {
                "answer": answer,
                "payload": payload,
            }

            return (
                answer,
                payload["overview"],
                payload["attention"]["markdown"],
                payload["softmax"]["markdown"],
                payload["layers"]["markdown"],
                payload["theory"]["overview"],
                payload["theory"]["feedback"],
                payload["theory"]["attention"],
                payload["theory"]["softmax"],
                payload["theory"]["layers"],
                "_Click “That’s Wrong” to see how feedback reshapes the internals._",
                "<em>No comparison yet.</em>",
                "<em>No comparison yet.</em>",
                "<em>No comparison yet.</em>",
            )

        def on_wrong_clicked(question):
            """Handle 'That's Wrong!' button - tell model its answer was wrong."""
            if not question or question not in original_answer_cache:
                return (
                    "⚠️ Please generate an answer first!",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            original_entry = original_answer_cache[question]
            original_answer = original_entry["answer"]
            original_payload = original_entry["payload"]

            correction_context = f"{original_answer} is wrong, the right answer is"
            corrected_answer, corrected_payload = generate_one_word_answer(question, context=correction_context)

            if isinstance(corrected_payload, str):
                return (
                    corrected_payload,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            summary, attention_html, softmax_html, layers_html = build_correction_sections(
                question,
                original_answer,
                original_payload,
                correction_context,
                corrected_answer,
                corrected_payload,
            )

            return summary, attention_html, softmax_html, layers_html

        def on_correction(question, correction):
            """Handle manual correction with specific answer."""
            if not correction.strip():
                return (
                    "⚠️ Please enter a correction!",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            if not question or question not in original_answer_cache:
                return (
                    "⚠️ Please generate an answer first!",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            original_entry = original_answer_cache[question]
            original_answer = original_entry["answer"]
            original_payload = original_entry["payload"]

            correction_context = f"{original_answer} is wrong, the right answer is {correction.strip()}"
            corrected_answer, corrected_payload = generate_one_word_answer(question, context=correction_context)

            if isinstance(corrected_payload, str):
                return (
                    corrected_payload,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            summary, attention_html, softmax_html, layers_html = build_correction_sections(
                question,
                original_answer,
                original_payload,
                correction_context,
                corrected_answer,
                corrected_payload,
            )

            return summary, attention_html, softmax_html, layers_html

        generate_btn.click(
            fn=on_generate,
            inputs=[question_input],
            outputs=[
                response_output,
                query_overview_md,
                query_attention_panel,
                query_softmax_panel,
                query_layers_panel,
                theory_overview_md,
                theory_feedback_md,
                theory_attention_md,
                theory_softmax_md,
                theory_layers_md,
                correction_summary_md,
                correction_attention_panel,
                correction_softmax_panel,
                correction_layers_panel,
            ],
            show_progress="full"
        )

        wrong_btn.click(
            fn=on_wrong_clicked,
            inputs=[question_input],
            outputs=[
                correction_summary_md,
                correction_attention_panel,
                correction_softmax_panel,
                correction_layers_panel,
            ],
            show_progress="full"
        )

        correction_btn.click(
            fn=on_correction,
            inputs=[question_input, correction_input],
            outputs=[
                correction_summary_md,
                correction_attention_panel,
                correction_softmax_panel,
                correction_layers_panel,
            ],
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
