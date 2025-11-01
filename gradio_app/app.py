"""
LLM Inference Tracker - Gradio App

Track how LLMs change their responses when corrected, with visualizations
of internal mechanisms: attention, softmax, and layer-by-layer predictions.

Deployment: Hugging Face Spaces
"""

import html
import re

import gradio as gr
from typing import List, Tuple, Dict, Set

from backend.llm_with_internals import LLMWithInternals
from visualizations.answer_flow import get_clean_words, clean_token


# Global model instance (loaded once)
MODEL = None


def load_model():
    """Load model on first use (lazy loading)."""
    global MODEL
    if MODEL is None:
        print("🔧 Loading TinyLlama model...")
        MODEL = LLMWithInternals(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("✅ Model ready!")
    return MODEL


def _format_percentage_bar(
    label: str, value: float, color_var: str, secondary_label: str = ""
) -> str:
    """Create a horizontal bar for percentage-based metrics."""
    safe_label = html.escape(label)
    safe_secondary = html.escape(secondary_label) if secondary_label else "&nbsp;"
    display_pct = round(value, 1)
    safe_value = max(0.0, value)
    bar_width = min(100, int(round(safe_value)))
    secondary_html = f"<span class=\"metric-secondary\">{safe_secondary}</span>"
    return "".join(
        [
            "<div class=\"metric-row\">",
            f"<span class=\"metric-label\">{safe_label}</span>",
            secondary_html,
            "<div class=\"metric-track\">",
            f"<div class=\"metric-fill\" style=\"width: {bar_width}%; background: var({color_var});\"></div>",
            "</div>",
            f"<span class=\"metric-value\">{display_pct}%</span>",
            "</div>",
        ]
    )


def _collect_attention_metrics(internals: dict) -> Tuple[List[Tuple[str, float]], str]:
    """Return attention percentages for every prompt/context word with formatted HTML."""
    attention_percentages = internals.get("attention_percentages")
    if attention_percentages is None:
        return [], "_Attention data is not available for this turn yet._"

    input_tokens = internals.get("input_tokens", [])
    question = internals.get("question", "")
    context = internals.get("context", "")
    if not question and not context:
        return [], "_Provide a prompt to see attention percentages._"

    def split_words(text: str) -> List[str]:
        if not text:
            return []
        return [word for word in re.split(r"[^0-9A-Za-z']+", text) if word]

    question_words = split_words(question)
    context_words = split_words(context)

    word_sources: Dict[str, Set[str]] = {}
    display_lookup: Dict[str, str] = {}
    ordered_keys: List[str] = []

    for word in question_words:
        lowered = word.lower()
        if not lowered:
            continue
        if lowered not in word_sources:
            ordered_keys.append(lowered)
            display_lookup[lowered] = word
            word_sources[lowered] = set()
        word_sources[lowered].add("Prompt")

    for word in context_words:
        lowered = word.lower()
        if not lowered:
            continue
        if lowered not in word_sources:
            ordered_keys.append(lowered)
            display_lookup[lowered] = word
            word_sources[lowered] = set()
        word_sources[lowered].add("Context")

    if not ordered_keys:
        return [], "_No prompt words were captured for attention tracking._"

    word_token_map: Dict[str, Dict[str, List[int]]] = {}
    for word, token_indices in get_clean_words(input_tokens):
        lowered = word.lower().strip()
        if not lowered:
            continue
        entry = word_token_map.setdefault(lowered, {"display": word, "indices": []})
        entry["indices"].extend(token_indices)

    metrics: List[Tuple[str, float]] = []
    metric_rows: List[str] = []

    prompt_html = "<div class=\"prompt-summary\">"
    prompt_html += f"<strong>Prompt:</strong> {html.escape(question) if question else '—'}"
    if context:
        prompt_html += f"<br><strong>Context:</strong> {html.escape(context)}"
    prompt_html += "</div>"

    for lowered in ordered_keys:
        entry = word_token_map.get(lowered)
        display_word = entry["display"] if entry else display_lookup.get(lowered, lowered)
        total_attention = 0.0
        if entry:
            for idx in entry["indices"]:
                if idx < len(attention_percentages):
                    total_attention += float(attention_percentages[idx])
        sources = word_sources.get(lowered, {"Prompt"})
        if len(sources) == 2:
            source_label = "Prompt + Context"
        else:
            source_label = next(iter(sources))
        metrics.append((display_word, total_attention))
        metric_rows.append(
            _format_percentage_bar(display_word, total_attention, "--attention-color", source_label)
        )

    metrics_html = "<div class=\"metric-list\">" + "".join(metric_rows) + "</div>"
    return metrics, prompt_html + metrics_html


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
        raw_logit = item.get("logit")
        logit = float(raw_logit) if raw_logit is not None else None
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
        bar_width = min(100, int(round(prob)))
        logit_text = f"logit {logit:+.2f}" if logit is not None else "—"
        rows.append(
            "<div class=\"metric-row\">"
            f"<span class=\"metric-label\">{token_display}</span>"
            f"<span class=\"metric-logit\">{logit_text}</span>"
            "<div class=\"metric-track\">"
            f"<div class=\"metric-fill\" style=\"width: {bar_width}%; background: var(--softmax-color);\"></div>"
            "</div>"
            f"<span class=\"metric-value\">{prob:.1f}%</span>"
            "</div>"
        )

    total_prob = sum(entry["prob"] for entry in items)
    remainder = max(0.0, 100.0 - total_prob)
    if remainder > 0.05:
        bar_width = min(100, int(round(remainder)))
        items.append({
            "token": "Other tokens",
            "prob": remainder,
            "logit": None,
        })
        rows.append(
            "<div class=\"metric-row\">"
            "<span class=\"metric-label\">Other tokens</span>"
            "<span class=\"metric-logit\">—</span>"
            "<div class=\"metric-track\">"
            f"<div class=\"metric-fill\" style=\"width: {bar_width}%; background: var(--softmax-color);\"></div>"
            "</div>"
            f"<span class=\"metric-value\">{remainder:.1f}%</span>"
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

<div class=\"section-note pill-softmax\"><strong>High-level flow:</strong> a question is turned into tokens, each pass through the network refines their meaning, and the model finally chooses the next word by comparing many possibilities.</div>

1. **Embedding & position sense.** The text is split into sub-word pieces and turned into vectors that encode meaning plus order.
2. **Attention exchanges.** Each layer lets tokens borrow context from one another, so the model can pick out the words that really matter.
3. **Layer updates.** Feed-forward blocks transform the focused information into sharper internal features.
4. **Prediction head.** The refined vector is scored against every vocabulary item, then softmax produces a clean probability distribution.

This app surfaces the three most important snapshots: attention weights (what it focused on), the softmax slate (which words competed), and the layer-by-layer probes (how certainty built up).
""".strip(),
    "feedback": """
### What happens when you correct the model

<div class=\"section-note pill-layers\"><strong>Why feedback matters:</strong> telling the model it is wrong nudges the internal state on the next run. The prompt now includes your hint, so attention, layer activations, and softmax scores shift toward the correction.</div>

* **Attention.** The reminder phrase pulls focus toward the corrective words, so the model rereads the prompt with that constraint in mind.
* **Layer dynamics.** Early layers incorporate the new clue, while deeper layers propagate it, pushing the correct answer higher.
* **Softmax.** Competing tokens lose probability mass as the corrected token gathers stronger evidence.

Even though weights are frozen, the conversation context acts like temporary memory: every new turn recomputes the internals using both the original question and your feedback.
""".strip(),
    "attention": """
### How attention works

<div class=\"section-note pill-attention\"><strong>Plain-language intuition:</strong> attention assigns a weight to each input word so the model can concentrate on the parts of the question that matter most for the next word it will say.</div>

**Formula:**

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

Step-by-step:

1. Compare the current prediction vector (**Q**) with every word from the prompt (**K**) to see which ones feel related.
2. Scale the scores by $1/\\sqrt{d_k}$ so large vectors do not explode the numbers.
3. Use softmax to translate those scores into easy-to-read percentages.
4. Blend the value vectors (**V**) using those percentages, giving the model a context summary tailored to the next word.

The orange bars in the Query tab show which words grabbed the most weight for the final answer.
""".strip(),
    "softmax": """
### How softmax turns scores into probabilities

<div class=\"section-note pill-softmax\"><strong>Plain-language intuition:</strong> the model assigns a raw score to every possible next token. Softmax stretches those scores so that they add up to 100%, making it easy to see which word is most likely.</div>

**Formula:**

$$\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

What this means in practice:

* Each candidate token gets exponentiated ($e^{z_i}$), exaggerating stronger evidence.
* All exponentiated scores are normalised by their sum so they form a true probability distribution.
* Even small score gaps can create big probability differences, which is why the purple bars can look decisive.

Sampling or temperature tweaks would flatten or sharpen these probabilities, but here we keep it simple so the behaviour stays interpretable.
""".strip(),
    "layers": """
### Why layer-by-layer probes are helpful

<div class=\"section-note pill-layers\"><strong>Plain-language intuition:</strong> each transformer layer refines the hidden representation of the answer. Probing with the logit lens shows how confidence in the final word grows as we move deeper into the network.</div>

TinyLlama has 22 transformer layers. For every layer we temporarily plug in the final prediction head and read the implied probabilities:

* **Early layers** capture broad topics (“this is about geography”). The correct word may only have a small lead.
* **Middle layers** integrate attention results and rule out conflicting options.
* **Late layers** sharpen the concept until one token clearly dominates.

When feedback is applied, look for the blue cards to show the corrected token rising sooner or more steeply across the stack.
""".strip(),
}


def build_visualization_payload(internals: dict, guidance_note: str = "") -> dict:
    """Bundle friendly explanations, metrics, and theory snippets for the UI."""

    question = internals.get("question", "").strip() or "your question"
    answer = internals.get("response", "").strip() or internals.get("text", "")

    overview_lines = [
        "### How the model responded",
        f"It read **{html.escape(question)}** and proposed **{html.escape(answer or '...')}** as its next word.",
        "The tabs below unpack where it placed attention, how likely each candidate word looked, and how certainty grew across layers.",
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

    def strip_section_note(markup: str) -> str:
        if not markup:
            return "<em>No data available.</em>"
        if "section-note" not in markup:
            return markup
        closing = markup.find("</div>")
        if closing == -1:
            return markup
        trimmed = markup[closing + len("</div>"):].strip()
        return trimmed or "<em>No data available.</em>"

    def extract_markdown(payload: dict, key: str) -> str:
        if not isinstance(payload, dict):
            return "<em>No data available.</em>"
        section = payload.get(key, {})
        if not isinstance(section, dict):
            return "<em>No data available.</em>"
        markup = section.get("markdown", "")
        if not markup:
            return "<em>No data available.</em>"
        return strip_section_note(markup)

    safe_question = html.escape(question)
    safe_original = html.escape(original_answer or "—")
    safe_corrected = html.escape(corrected_answer or "—")
    safe_context = html.escape(correction_context)

    summary = f"""
<div class="section-note pill-layers">
    <strong>Question:</strong> {safe_question}<br>
    <strong>Before feedback:</strong> {safe_original}<br>
    <strong>After feedback:</strong> {safe_corrected}
</div>
<p style="font-size:0.9rem;color:#4B5563;">We reminded the model: <em>{safe_context}</em></p>
"""

    attention = f"""
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

    softmax = f"""
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

    layers = f"""
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

        :root {
            --accent-blue: #2563EB;
            --accent-teal: #14B8A6;
            --surface-blue: rgba(37, 99, 235, 0.04);
            --surface-blue-strong: rgba(37, 99, 235, 0.08);
            --surface-teal: rgba(20, 184, 166, 0.08);
            --attention-color: #F97316;
            --attention-bg: rgba(249, 115, 22, 0.12);
            --softmax-color: #6366F1;
            --softmax-bg: rgba(99, 102, 241, 0.12);
            --layers-color: #0EA5E9;
            --layers-bg: rgba(14, 165, 233, 0.12);
        }

        body, input, textarea, button {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        body {
            font-size: 15px;
            line-height: 1.65;
            color: #111827;
            background: #ffffff;
        }

        p, li {
            line-height: 1.7;
        }

        /* Minimalistic elegant styling */
        .container {
            max-width: 960px;
            margin: 0 auto;
        }

        .title-section {
            text-align: center;
            padding: 0.85rem 1.75rem 1.4rem 1.75rem;
            margin-bottom: 1.75rem;
            background: linear-gradient(180deg, rgba(37, 99, 235, 0.06) 0%, rgba(255, 255, 255, 0) 100%);
            border-radius: 8px;
        }

        .title-main {
            font-size: 1.9rem;
            font-weight: 400;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin: 0 0 0.75rem 0;
        }

        .title-subtitle {
            font-size: 1rem;
            font-weight: 400;
            color: #4b5563;
            margin: 0;
            line-height: 1.7;
            max-width: 620px;
            margin: 0 auto;
        }

        .prompt-summary {
            text-align: left;
            font-size: 0.9rem;
            color: #1f2937;
            background: rgba(15, 23, 42, 0.03);
            border-radius: 6px;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
        }

        .prompt-summary strong {
            color: #0f172a;
        }

        .guide-inline {
            background: var(--surface-blue);
            padding: 1.35rem 1.5rem;
            margin: 1.5rem auto;
            font-size: 0.95rem;
            color: #1f2937;
            line-height: 1.75;
            max-width: 840px;
            border-left: 3px solid var(--accent-blue);
            border-radius: 6px;
            box-shadow: inset 0 1px 2px rgba(37, 99, 235, 0.08);
        }

        .guide-inline p {
            margin: 0 0 0.5rem 0;
        }

        .guide-inline ol {
            margin: 0.25rem 0 0 1.25rem;
            padding: 0;
        }

        .guide-inline li {
            margin-bottom: 0.35rem;
        }

        .section-divider {
            height: 1px;
            background: rgba(15, 23, 42, 0.08);
            margin: 1.75rem 0;
        }

        /* Smaller buttons */
        button {
            padding: 0.55rem 1.35rem !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            border-radius: 4px !important;
            letter-spacing: 0.01em;
        }

        button.secondary {
            color: #0f172a !important;
        }

        /* Clean inputs */
        input, textarea {
            border: 1px solid rgba(15, 23, 42, 0.12) !important;
            border-radius: 4px !important;
            font-size: 0.95rem !important;
            background: rgba(255, 255, 255, 0.9) !important;
        }

        /* Theory boxes */
        .theory-box {
            background: linear-gradient(180deg, var(--surface-teal) 0%, rgba(255, 255, 255, 0) 100%);
            border-left: 3px solid var(--accent-teal);
            padding: 1.75rem;
            margin: 1.75rem 0;
            border-radius: 6px;
        }

        .correction-guide {
            background: linear-gradient(180deg, rgba(37, 99, 235, 0.05) 0%, rgba(255, 255, 255, 0) 100%);
            border-radius: 8px;
            border: 1px solid rgba(37, 99, 235, 0.15);
            padding: 1.75rem;
            margin-top: 2.5rem;
        }

        .correction-guide h4 {
            margin: 0 0 0.75rem 0;
            font-size: 1.05rem;
            color: #1f2937;
        }

        .correction-guide ul {
            margin: 0.75rem 0 0 1.25rem;
            color: #334155;
        }

        .comparison-table thead th {
            background: rgba(15, 23, 42, 0.04);
        }

        .comparison-table td {
            background: rgba(255, 255, 255, 0.92);
        }

        .comparison-table tr:not(:last-child) td {
            border-bottom: 1px solid rgba(15, 23, 42, 0.08);
        }

        .soft-note {
            margin-top: 1rem;
            padding: 0.75rem 1rem;
            background: #F9FAFB;
            border-left: 3px solid var(--accent-blue);
            color: #1F2937;
            font-size: 0.9rem;
            border-radius: 4px;
        }

        .section-note {
            padding: 0.85rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
            line-height: 1.6;
        }

        .pill-attention {
            background: var(--attention-bg);
            border-left: 3px solid var(--attention-color);
        }

        .pill-softmax {
            background: var(--softmax-bg);
            border-left: 3px solid var(--softmax-color);
        }

        .pill-layers {
            background: var(--layers-bg);
            border-left: 3px solid var(--layers-color);
        }

        .metric-list {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: 160px 90px 1fr 60px;
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
            color: #6B7280;
        }

        .metric-secondary {
            font-size: 0.7rem;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-track {
            position: relative;
            width: 100%;
            height: 6px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.08);
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
            color: #4B5563;
            font-weight: 500;
        }

        .layer-grid {
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }

        .layer-card {
            border: 1px solid rgba(15, 23, 42, 0.1);
            border-radius: 10px;
            padding: 1rem;
            background: #FFFFFF;
            box-shadow: 0 4px 8px rgba(15, 23, 42, 0.04);
        }

        .layer-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--layers-color);
            margin-bottom: 0.5rem;
        }

        .layer-actual {
            font-size: 0.9rem;
            color: #111827;
            margin-bottom: 0.5rem;
        }

        .layer-alternatives {
            margin: 0;
            padding-left: 1rem;
            color: #4B5563;
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
            border: 1px solid rgba(15, 23, 42, 0.1);
            border-radius: 10px;
            padding: 1rem;
            background: #FFFFFF;
        }
        """
    ) as demo:

        gr.HTML("""
        <div class="title-section">
            <h1 class="title-main">LLM under the hood</h1>
            <p class="title-subtitle">
                Follow a compact transformer as it reads your question, weighs every word, and settles on a one-word reply.<br>
                Use the panels below to watch how attention, probabilities, and layers react when you correct it.
            </p>
        </div>
        """)

        gr.HTML("""
        <div class="guide-inline">
            <p><strong>How to explore:</strong> Ask a short factual question whose answer should be a single word, then study how the model handled it in each tab.</p>
            <p><strong>Recommended flow:</strong></p>
            <ol>
                <li>Run a question such as "What color is banana?" and note the answer.</li>
                <li>Inspect the Query tabs to see which words mattered, which candidates competed, and how confidence grew.</li>
                <li>Use <em>That's Wrong</em> or add the true word, then compare how the internals shift after feedback.</li>
            </ol>
            <p><strong>Model:</strong> TinyLlama-1.1B (22 transformer layers, ~2-5 seconds per response)</p>
        </div>
        """)

        gr.HTML('<div class="section-divider"></div>')

        empty_attention_html = "<em>Run the model to list attention percentages for every word in your prompt and context.</em>"
        empty_softmax_html = "<em>Run the model to reveal the top candidate words and their probabilities.</em>"
        empty_layers_html = "<em>Run the model to chart how confidence evolves across layers.</em>"
        correction_placeholder = "_Generate an answer and press “That’s Wrong” to compare before and after._"

        with gr.Tabs():
            with gr.Tab("Query"):
                gr.Markdown("### Ask the model a quick fact")
                gr.Markdown(
                    "Use this tab to follow the model's first attempt. Short prompts such as **What is the capital of Australia?** work best because the answer should be a single word."
                )

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

                query_overview_md = gr.Markdown(
                    "_Run the model to see how it read your prompt, picked a candidate, and justified the choice._"
                )

                with gr.Tabs():
                    with gr.Tab("🟠 Attention"):
                        gr.Markdown(
                            "Every word from your prompt and any added context appears here with an attention percentage so you can see what the model reread."
                        )
                        query_attention_panel = gr.HTML(empty_attention_html)

                    with gr.Tab("🟣 Softmax"):
                        gr.Markdown(
                            "Purple bars reveal the probability of the highest-scoring candidate words just before the model spoke."
                        )
                        query_softmax_panel = gr.HTML(empty_softmax_html)

                    with gr.Tab("🔵 Layer by Layer"):
                        gr.Markdown(
                            "Blue cards track how confidence in the final word grows as information flows through deeper transformer layers."
                        )
                        query_layers_panel = gr.HTML(empty_layers_html)

            with gr.Tab("Correction"):
                gr.Markdown("### Teach the model when it slips")
                gr.Markdown(
                    "Challenge the answer and optionally provide the correct word. The comparison panels show exactly how the internal signals respond to your feedback."
                )

                wrong_btn = gr.Button("That's Wrong", variant="stop", size="sm")

                gr.Markdown("**Optional:** Type the correct word so the model can aim directly at it.")
                correction_input = gr.Textbox(
                    label="",
                    placeholder="Type the word it should have said",
                    lines=1,
                    show_label=False
                )

                correction_btn = gr.Button("Submit Correction", variant="secondary", size="sm")

                correction_summary_md = gr.Markdown(correction_placeholder)

                with gr.Tabs():
                    with gr.Tab("🟠 Attention"):
                        gr.Markdown("See how focus on each prompt or context word changes once the correction is in play.")
                        correction_attention_panel = gr.HTML("<em>No comparison yet.</em>")

                    with gr.Tab("🟣 Softmax"):
                        gr.Markdown("Check whether the corrected word overtakes its competitors in the probability slate.")
                        correction_softmax_panel = gr.HTML("<em>No comparison yet.</em>")

                    with gr.Tab("🔵 Layer by Layer"):
                        gr.Markdown("Look for deeper layers to embrace the corrected answer sooner or with greater certainty.")
                        correction_layers_panel = gr.HTML("<em>No comparison yet.</em>")

            with gr.Tab("Theory"):
                gr.Markdown("### Peek behind the math")
                gr.Markdown(
                    "These notes explain the coloured panels in everyday language before linking them back to the underlying equations."
                )

                theory_overview_md = gr.Markdown(THEORY_TEXT["overview"])
                theory_feedback_md = gr.Markdown(THEORY_TEXT["feedback"])

                with gr.Tabs():
                    with gr.Tab("🟠 Attention"):
                        theory_attention_md = gr.Markdown(THEORY_TEXT["attention"])

                    with gr.Tab("🟣 Softmax"):
                        theory_softmax_md = gr.Markdown(THEORY_TEXT["softmax"])

                    with gr.Tab("🔵 Layer by Layer"):
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
