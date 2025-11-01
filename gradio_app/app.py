"""
LLM Inference Tracker - Gradio App

Track how LLMs change their responses when corrected, with visualizations
of internal mechanisms: attention, softmax, and layer-by-layer predictions.

Deployment: Hugging Face Spaces
"""

import html
import math
import re

import gradio as gr
from typing import List, Tuple, Dict, Optional

from backend.llm_with_internals import LLMWithInternals
from visualizations.answer_flow import get_clean_words, clean_token


# Global model instance (loaded once)
MODEL = None
DEFAULT_TEMPERATURE = 0.2


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


def _prepare_prompt_word_rows(internals: dict) -> Tuple[List[dict], str, Optional[str]]:
    """Return ordered prompt/context words aligned with their token indices."""

    def split_words(text: str) -> List[str]:
        if not text:
            return []
        return [word for word in re.split(r"[^0-9A-Za-z']+", text) if word]

    def normalise_word(text: str) -> str:
        return re.sub(r"[^0-9A-Za-z']+", "", (text or "")).lower()

    question = internals.get("question", "")
    context = internals.get("context", "")
    prompt_html = "<div class=\"prompt-summary\">"
    prompt_html += f"<strong>Prompt:</strong> {html.escape(question) if question else '—'}"
    if context:
        prompt_html += f"<br><strong>Context:</strong> {html.escape(context)}"
    prompt_html += "</div>"

    if not question and not context:
        return [], prompt_html, "_Provide a prompt to see attention percentages._"

    ordered_words: List[dict] = []
    for source, raw_text in (("Prompt", question), ("Context", context)):
        for word in split_words(raw_text):
            ordered_words.append(
                {
                    "label": word,
                    "key": normalise_word(word),
                    "source": source,
                }
            )

    if not ordered_words:
        return [], prompt_html, "_No prompt words were captured for attention tracking._"

    token_assignments: List[List[int]] = [None] * len(ordered_words)  # type: ignore
    prompt_tokens = internals.get("input_tokens", [])
    cleaned_tokens = list(get_clean_words(prompt_tokens))

    for word, token_indices in cleaned_tokens:
        key = normalise_word(word)
        if not key:
            continue
        for idx, entry in enumerate(ordered_words):
            if entry["key"] != key:
                continue
            if token_assignments[idx] is None:
                token_assignments[idx] = list(token_indices)
                break

    rows: List[dict] = []
    for idx, entry in enumerate(ordered_words):
        rows.append(
            {
                "order": idx,
                "label": entry["label"],
                "source": entry["source"],
                "token_indices": token_assignments[idx] or [],
            }
        )

    return rows, prompt_html, None


def _collect_attention_metrics(internals: dict) -> Tuple[List[Tuple[str, float]], str]:
    """Return attention percentages for every prompt/context word with formatted HTML."""

    attention_percentages = internals.get("attention_percentages")
    if attention_percentages is None:
        return [], "_Attention data is not available for this turn yet._"

    rows, prompt_html, warning = _prepare_prompt_word_rows(internals)
    if warning:
        return [], prompt_html + warning

    metrics: List[Tuple[str, float]] = []
    metric_rows: List[str] = []

    for row in rows:
        total_attention = 0.0
        for idx in row["token_indices"]:
            if idx < len(attention_percentages):
                total_attention += float(attention_percentages[idx])
        source_label = row["source"]
        metrics.append((row["label"], total_attention))
        metric_rows.append(
            _format_percentage_bar(row["label"], total_attention, "--attention-color", source_label)
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


def _display_token(token: str) -> str:
    cleaned = clean_token(token or "")
    if cleaned:
        token_text, _ = cleaned
        token_text = token_text.strip()
        if token_text:
            return html.escape(token_text)
    safe = (token or "").replace("▁", " ").strip()
    return html.escape(safe or token or "∅")


def _render_attention_heatmap(heatmap: dict, internals: dict) -> str:
    if not heatmap:
        return "<em>Run the model to see a token-by-token attention heatmap.</em>"

    raw_output_tokens = heatmap.get("output_tokens") or []
    values = heatmap.get("values") or []

    if not raw_output_tokens or not values:
        return "<em>Attention heatmap data is unavailable.</em>"

    matched_rows, _, warning = _prepare_prompt_word_rows(
        {
            **internals,
            "input_tokens": heatmap.get("prompt_tokens") or [],
        }
    )

    if warning and not matched_rows:
        return warning.replace("attention percentages", "attention heatmap")

    if not matched_rows:
        return "<em>No prompt or context words were detected for the heatmap.</em>"

    matched_rows.sort(key=lambda item: item["order"])

    # Combine output tokens into readable words.
    output_word_entries = []
    output_words = get_clean_words(raw_output_tokens)
    if output_words:
        for idx, (word, token_indices) in enumerate(output_words):
            output_word_entries.append(
                {
                    "label": word,
                    "indices": token_indices,
                    "order": idx,
                }
            )
    else:
        for idx, token in enumerate(raw_output_tokens):
            output_word_entries.append(
                {
                    "label": (_display_token(token) or f"Token {idx+1}"),
                    "indices": [idx],
                    "order": idx,
                }
            )

    if not output_word_entries or not values:
        return "<em>Attention heatmap data is unavailable.</em>"

    header_cells = []
    for entry in output_word_entries:
        header_cells.append(f"<th>{html.escape(entry['label'])}</th>")

    rows_html = []
    for row in matched_rows:
        cells = []
        for column in output_word_entries:
            pct = 0.0
            for output_idx in column["indices"]:
                if output_idx >= len(values):
                    continue
                column_values = values[output_idx]
                if not column_values:
                    continue
                if row["token_indices"]:
                    for prompt_idx in row["token_indices"]:
                        if prompt_idx < len(column_values):
                            pct += float(column_values[prompt_idx])
                else:
                    # Row without explicit tokens still appears but registers zero attention.
                    pct += 0.0
            intensity = min(0.85, 0.15 + (pct / 100.0) * 0.7)
            cells.append(
                "<td style=\"background: rgba(249, 115, 22, {intensity:.3f});\">"
                f"<span class=\"heatmap-value\">{pct:.1f}%</span>"
                "</td>".format(intensity=intensity)
            )

        source_badge = f"<span class=\"heatmap-source heatmap-source-{row['source'].lower()}\">{row['source']}</span>"
        rows_html.append(
            "<tr>"
            f"<th scope=\"row\">{source_badge}{html.escape(row['label'])}</th>"
            + "".join(cells)
            + "</tr>"
        )

    header_html = "<tr><th>Prompt &amp; context words</th>" + "".join(header_cells) + "</tr>"

    return (
        "<div class=\"heatmap-wrapper\">"
        "<div class=\"heatmap-heading\"><strong>Attention heatmap</strong> — each cell shows how strongly an output word re-read your prompt or context terms.</div>"
        "<table class=\"heatmap-table\">"
        + header_html
        + "".join(rows_html)
        + "</table>"
        "</div>"
    )


def _render_softmax_waterfall(metrics: List[dict], temperature: Optional[float] = None) -> str:
    if not metrics:
        return "<em>Softmax waterfall will appear after the model proposes a token.</em>"

    logit_entries = [m for m in metrics if m.get("logit") is not None]
    if not logit_entries:
        return "<em>Logit data was unavailable for this token.</em>"

    max_logit = max(m["logit"] for m in logit_entries)
    sum_shifted_all = None
    shifted_values = {}

    for entry in logit_entries:
        prob = max(entry.get("prob", 0.0) / 100.0, 1e-9)
        shift = entry["logit"] - max_logit
        exp_shift = math.exp(shift)
        shifted_values[entry["token"]] = exp_shift
        if sum_shifted_all is None and prob > 0:
            sum_shifted_all = exp_shift / prob

    if sum_shifted_all is None:
        sum_shifted_all = sum(shifted_values.values()) or 1.0

    rows = []
    for entry in metrics:
        token = entry.get("token", "?")
        prob_pct = float(entry.get("prob", 0.0))
        prob_fraction = prob_pct / 100.0

        if entry.get("logit") is not None:
            exp_shift = shifted_values.get(token, 0.0)
        else:
            exp_shift = prob_fraction * sum_shifted_all

        exp_width = min(100, int(round((exp_shift / sum_shifted_all) * 100))) if sum_shifted_all else 0
        prob_width = min(100, int(round(prob_pct)))
        logit_text = f"{entry['logit']:+.2f} logit" if entry.get("logit") is not None else "—"

        rows.append(
            "<div class=\"waterfall-row\">"
            f"<div class=\"waterfall-token\">{html.escape(token)}</div>"
            f"<div class=\"waterfall-stage\">{logit_text}</div>"
            f"<div class=\"waterfall-stage\"><div class=\"waterfall-bar exp\" style=\"width:{exp_width}%;\"></div><span>exp= {exp_shift:.3f}</span></div>"
            f"<div class=\"waterfall-stage\"><div class=\"waterfall-bar prob\" style=\"width:{prob_width}%;\"></div><span>{prob_pct:.1f}%</span></div>"
            "</div>"
        )

    if temperature is not None:
        explainer = (
            "<div class=\"waterfall-heading\"><strong>Softmax waterfall</strong> — logits are scaled by the temperature "
            f"(<code>{temperature:.2f}</code>) before exponentiation. Higher temperatures flatten the exp column; lower ones make the winning bar sharper.</div>"
        )
    else:
        explainer = (
            "<div class=\"waterfall-heading\"><strong>Softmax waterfall</strong> — track each token from raw logit through exponentiation to the final probability.</div>"
        )

    explainer += (
        "<p class=\"waterfall-note\">Read the columns left to right: the first shows the raw logit (after temperature), the middle shows the exponentiated value before normalisation, and the final purple bar is the true probability that feeds sampling.</p>"
    )

    return (
        "<div class=\"waterfall-wrapper\">"
        + explainer
        + "".join(rows)
        + "</div>"
    )


def _render_layer_sparkline(metrics: List[dict], generated_tokens: List[str]) -> str:
    if not metrics:
        return "<em>Run the model to chart confidence growth across layers.</em>"

    confidences = [max(0.0, min(100.0, (entry.get("actual") or {}).get("prob", 0.0))) for entry in metrics]
    if not any(confidences):
        return "<em>Confidence sparkline will appear once the model locks onto a candidate.</em>"

    width = 360
    height = 120
    padding = 24
    step = (width - 2 * padding) / max(1, len(confidences) - 1)
    points = []
    for idx, prob in enumerate(confidences):
        x = padding + step * idx
        y = height - padding - (prob / 100.0) * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")

    area_points = " ".join([f"{padding},{height - padding}"] + points + [f"{padding + step * (len(confidences) - 1):.1f},{height - padding}"])
    polyline_points = " ".join(points)
    final_token = generated_tokens[0] if generated_tokens else "the answer"
    axis_y = height - padding
    axis_end = padding + step * max(1, len(confidences) - 1)
    tick_marks = []
    for idx in range(len(confidences)):
        x = padding + step * idx
        tick_marks.append(
            f"<line x1='{x:.1f}' y1='{axis_y}' x2='{x:.1f}' y2='{axis_y + 6}' class='sparkline-tick-mark' />"
        )
        tick_marks.append(
            f"<text x='{x:.1f}' y='{axis_y + 18}' class='sparkline-tick-label'>{idx}</text>"
        )
    ticks_svg = "".join(tick_marks)

    return (
        "<div class=\"sparkline-wrapper\">"
        "<div class=\"sparkline-heading\"><strong>Confidence sparkline</strong> — see how the model warmed up to"
        f" {_display_token(final_token)}.</div>"
        f"<svg viewBox=\"0 0 {width} {height}\" class=\"sparkline\">"
        f"<path d=\"M {area_points} Z\" class=\"sparkline-area\" />"
        f"<polyline points=\"{polyline_points}\" class=\"sparkline-line\" />"
        f"<line x1='{padding}' y1='{axis_y}' x2='{axis_end:.1f}' y2='{axis_y}' class=\"sparkline-axis-line\" />"
        f"<text x='{axis_end:.1f}' y='{axis_y + 32}' class=\"sparkline-axis-caption\">Layer →</text>"
        f"{ticks_svg}"
        "</svg>"
        "</div>"
    )


def _render_feedback_delta(original_payload: dict, corrected_payload: dict) -> str:
    if not original_payload or not corrected_payload:
        return "<em>Generate a correction to compare attention, probabilities, and layers.</em>"

    def build_lookup(metrics):
        lookup = {}
        if not metrics:
            return lookup
        for item in metrics:
            if isinstance(item, tuple) and len(item) == 2:
                label, value = item
            elif isinstance(item, dict):
                label = item.get("token")
                value = item.get("prob", 0.0)
            else:
                continue
            key = (label or "").strip().lower()
            if not key:
                continue
            lookup[key] = {"label": label, "value": float(value)}
        return lookup

    orig_attn = build_lookup(original_payload.get("attention", {}).get("metrics"))
    corr_attn = build_lookup(corrected_payload.get("attention", {}).get("metrics"))
    attn_items = []
    for key in set(orig_attn) | set(corr_attn):
        before = orig_attn.get(key, {"label": corr_attn.get(key, {}).get("label", key), "value": 0.0})
        after = corr_attn.get(key, {"label": before.get("label", key), "value": 0.0})
        delta = after["value"] - before["value"]
        attn_items.append({"label": after.get("label") or before.get("label") or key, "delta": delta})

    attn_items = [item for item in attn_items if abs(item["delta"]) > 0.1]
    attn_items.sort(key=lambda item: abs(item["delta"]), reverse=True)
    attn_html = "".join(
        f"<li><span class=\"delta-label\">{html.escape(item['label'])}</span><span class=\"delta-value\">{item['delta']:+.1f} pts</span></li>"
        for item in attn_items[:4]
    ) or "<li>No major shifts detected.</li>"

    orig_softmax = build_lookup(original_payload.get("softmax", {}).get("metrics"))
    corr_softmax = build_lookup(corrected_payload.get("softmax", {}).get("metrics"))
    softmax_items = []
    for key in set(orig_softmax) | set(corr_softmax):
        before = orig_softmax.get(key, {"label": corr_softmax.get(key, {}).get("label", key), "value": 0.0})
        after = corr_softmax.get(key, {"label": before.get("label", key), "value": 0.0})
        delta = after["value"] - before["value"]
        softmax_items.append({"label": after.get("label") or before.get("label") or key, "delta": delta})

    softmax_items = [item for item in softmax_items if abs(item["delta"]) > 0.05]
    softmax_items.sort(key=lambda item: abs(item["delta"]), reverse=True)
    softmax_html = "".join(
        f"<li><span class=\"delta-label\">{html.escape(item['label'])}</span><span class=\"delta-value\">{item['delta']:+.1f} pts</span></li>"
        for item in softmax_items[:4]
    ) or "<li>Probability slate stayed steady.</li>"

    orig_layers = original_payload.get("layers", {}).get("metrics", [])
    corr_layers = corrected_payload.get("layers", {}).get("metrics", [])
    layer_changes = []
    for idx in range(max(len(orig_layers), len(corr_layers))):
        orig_entry = orig_layers[idx].get("actual") if idx < len(orig_layers) and isinstance(orig_layers[idx], dict) else None
        corr_entry = corr_layers[idx].get("actual") if idx < len(corr_layers) and isinstance(corr_layers[idx], dict) else None
        before = (orig_entry or {}).get("prob", 0.0)
        after = (corr_entry or {}).get("prob", 0.0)
        delta = after - before
        if abs(delta) < 0.1:
            continue
        layer_changes.append({"layer": idx, "delta": delta})

    layer_changes.sort(key=lambda item: abs(item["delta"]), reverse=True)
    layer_html = "".join(
        f"<li>Layer {item['layer']}: {item['delta']:+.1f} pts toward the answer</li>"
        for item in layer_changes[:4]
    ) or "<li>Layer confidence barely moved.</li>"

    return (
        "<div class=\"delta-wrapper\">"
        "<div class=\"delta-heading\"><strong>Feedback impact</strong> — quick view of what changed most.</div>"
        "<div class=\"delta-grid\">"
        f"<div class=\"delta-card\"><h4>Attention shifts</h4><ul>{attn_html}</ul></div>"
        f"<div class=\"delta-card\"><h4>Probability swings</h4><ul>{softmax_html}</ul></div>"
        f"<div class=\"delta-card\"><h4>Layer momentum</h4><ul>{layer_html}</ul></div>"
        "</div>"
        "</div>"
    )


PIPELINE_STEPS = [
    (
        "Tokenise prompt",
        "Split the question and any correction into tokens TinyLlama can read.",
    ),
    (
        "Share context via attention",
        "Each token looks back over the prompt to borrow the most relevant words.",
    ),
    (
        "Refine through transformer layers",
        "Feed-forward blocks reshape the focused context at every depth.",
    ),
    (
        "Score candidates",
        "The prediction head produces a logit score for every vocabulary option.",
    ),
    (
        "Sample the next token",
        "Softmax (plus the temperature slider) turns scores into the chosen answer.",
    ),
]


def _build_pipeline_html(
    completed_steps: int = 0,
    active_step: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    steps_html = []
    for index, (title, description) in enumerate(PIPELINE_STEPS, start=1):
        if index <= completed_steps:
            status_class = "pipeline-step-done"
            icon = "✓"
        elif active_step is not None and index == active_step:
            status_class = "pipeline-step-active"
            icon = "…"
        else:
            status_class = "pipeline-step-pending"
            icon = ""

        steps_html.append(
            "<li class=\"pipeline-step {status}\">".format(status=status_class)
            + f"<span class=\"pipeline-check\">{icon}</span>"
            + "<div class=\"pipeline-text\">"
            + f"<span class=\"pipeline-step-label\">{html.escape(title)}</span>"
            + f"<p>{html.escape(description)}</p>"
            + "</div>"
            + "</li>"
        )

    temperature_note = (
        f"<p class=\"pipeline-temperature\">Temperature: <strong>{temperature:.2f}</strong> — lower keeps answers deterministic, higher spreads probability mass.</p>"
        if temperature is not None
        else ""
    )

    return (
        "<div class=\"pipeline-card\">"
        "<h4>Transformer pipeline checklist</h4>"
        "<ol>"
        + "".join(steps_html)
        + "</ol>"
        + temperature_note
        + "</div>"
    )


THEORY_TEXT = {
    "overview": """
### Start with the transformer

Transformers treat language as a sequence of tokens. Each token becomes a vector, borrows context from its neighbours through self-attention, flows through stacked feed-forward blocks, and finally receives a score for how likely it is to appear next.

**Why TinyLlama?** TinyLlama-1.1B is an open chat model with about 1.1 billion parameters. That is far smaller than ChatGPT’s GPT-4, so responses can be rougher, but the compact size lets this app expose every attention map, logit, and hidden state for learning.

#### The pipeline you'll watch
1. **Tokenise & embed.** The question (and any correction) is split into tokens and turned into vectors with position information.
2. **Share context via attention.** Tokens look back over the prompt to borrow the words that matter most for the next guess.
3. **Refine through transformer layers.** Each of the 22 layers applies attention plus a feed-forward update to sharpen the meaning.
4. **Score candidates.** The final representation is compared with the vocabulary to produce a logit score for every possible next token.
5. **Sample the next token.** Softmax converts the logits to probabilities, optionally adjusted by the temperature slider, and the top token is emitted.

The checklist beside the title ticks through these stages as you run the model. The Query tab mirrors steps 2–5 in the orange attention view, the purple softmax panels, and the blue layer probes.

_Try it:_ ask for a factual answer such as **What is the capital of Australia?** Run it twice—first with a cool temperature (~0.2) and then with a warmer one (~1.0)—and watch how the checklist and visualisations change.
""".strip(),
    "feedback": """
### What happens when you give feedback

TinyLlama does not update its weights during this demo, but each new turn reuses the previous conversation. When you press a feedback button we prepend a short reminder to the next prompt, so the model recomputes every stage with your hint included.

1. **That's Wrong.** We add a gentle nudge that the previous answer missed the mark and let the model try again.
2. **Submit Correction.** We include the exact word you typed so attention, layers, and logits can aim directly at the expected answer.

During the rerun you will see:
* **Attention shifts.** Prompt words related to your feedback gather more weight.
* **Layer dynamics.** The corrected answer usually gains probability earlier in the layer stack.
* **Softmax rebalancing.** Competing tokens lose probability mass as the corrected token gathers evidence.

The Δ Impact tab summarises these changes so you can compare before vs after at a glance.

_Try it:_ generate an answer, press **That’s Wrong**, then inspect the Δ Impact tab. Next, type the correct word and submit a correction to see how the three panels evolve.
""".strip(),
    "attention": """
### How attention works

Attention lets each token decide which other tokens to reread before the model chooses the next word.

1. Build three vectors for every token: **query** (what I am looking for), **key** (how relevant I am), and **value** (the information to copy).
2. Compare each query with every key to get similarity scores, scale them to keep numbers stable, and apply softmax so they become percentages.
3. Use those percentages to blend the value vectors, giving the model a focused summary tailored to the next prediction.

The orange bar chart lists every word from your question or context with its total attention percentage. The heatmap underneath shows the same numbers broken down by output token so you can see which prompt words influenced each generated token.

_Try it:_ read the percentage printed in a heatmap cell, then find the same word in the bar chart to confirm that the totals line up.
""".strip(),
    "softmax": """
### How softmax turns scores into probabilities

After the transformer layers we have a logit score for each vocabulary token. Softmax translates those scores into a probability distribution that sums to 100%.

1. **Apply temperature.** Divide each logit by the slider value. Low temperatures (<1) sharpen differences; high ones (>1) flatten them.
2. **Exponentiate.** Apply $e^{z}$ so larger scores grow disproportionately.
3. **Normalise.** Divide by the sum of all exponentials to get the final probabilities.

The purple list shows the top candidates after normalisation. The waterfall view tracks the same tokens through the logit, exponential, and probability stages so you can see where the decisive jump happens. The “Other tokens” row sweeps up any remaining mass so everything adds to 100%.

_Try it:_ set the temperature to 0.2 and run a prompt, then set it to 1.0 and run again. Notice how the exponential column becomes more even at higher temperatures and how the probability bars spread out.
""".strip(),
    "layers": """
### Why layer-by-layer probes matter

A transformer builds its answer gradually as information flows through each layer. By applying the final prediction head at every layer (a technique called the logit lens) we can estimate how confident the model would be if it stopped early.

* **Early layers** capture broad topic hints, so the correct token may only lead slightly.
* **Middle layers** integrate attention results and suppress incompatible options.
* **Late layers** push the winning token far ahead, producing a near-certain probability.

The blue cards show the top alternatives at each layer, while the sparkline plots the probability of the actual answer with layer numbers labelled along the x-axis.

_Try it:_ after you submit a correction, look for the sparkline to bend upward sooner. That indicates the model embraced your hint earlier in the stack.
""".strip(),
}


def build_visualization_payload(internals: dict, guidance_note: str = "") -> dict:
    """Bundle friendly explanations, metrics, and theory snippets for the UI."""

    question = internals.get("question", "").strip() or "your question"
    answer = internals.get("response", "").strip() or internals.get("text", "")
    temperature = internals.get("temperature")

    overview_lines = [
        "### How the model responded",
        f"It read **{html.escape(question)}** and proposed **{html.escape(answer or '...')}** as its next word.",
        "The tabs below unpack where it placed attention, how likely each candidate word looked, and how certainty grew across layers.",
    ]
    if temperature is not None:
        if temperature <= 0.35:
            temp_descriptor = "a sharp, deterministic slate"
        elif temperature >= 0.9:
            temp_descriptor = "a very exploratory probability slate"
        else:
            temp_descriptor = "a balanced probability slate"
        overview_lines.append(
            f"Temperature was set to **{temperature:.2f}**, so expect {temp_descriptor}."
        )
    if guidance_note:
        overview_lines.append(f"<div class='soft-note'>{guidance_note}</div>")

    attention_metrics, attention_html = _collect_attention_metrics(internals)
    softmax_metrics, softmax_html = _collect_softmax_metrics(internals)
    layer_metrics, layer_html = _collect_layer_metrics(internals, answer)
    attention_heatmap_html = _render_attention_heatmap(internals.get("attention_heatmap"), internals)
    softmax_waterfall_html = _render_softmax_waterfall(
        softmax_metrics,
        internals.get("temperature"),
    )
    layer_sparkline_html = _render_layer_sparkline(layer_metrics, internals.get("generated_tokens", []))

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
            "heatmap": attention_heatmap_html,
        },
        "softmax": {
            "metrics": softmax_metrics,
            "markdown": (
                "<div class=\"section-note pill-softmax\"><strong>Top candidates:</strong> the probability shows how confident the model was before choosing.</div>"
                + softmax_html
            ),
            "waterfall": softmax_waterfall_html,
        },
        "layers": {
            "metrics": layer_metrics,
            "markdown": (
                "<div class=\"section-note pill-layers\"><strong>Confidence by layer:</strong> later layers should favour the final answer.</div>"
                + layer_html
            ),
            "sparkline": layer_sparkline_html,
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
) -> Tuple[str, str, str, str, str]:
    """Return summary, per-section comparison HTML blocks, and delta view."""

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
        <h4 style="margin-top:0; color: var(--attention-color);">Before feedback <span class="info-tooltip" data-tip="Attention percentages from the original prompt run.">?</span></h4>
        {extract_markdown(original_payload, 'attention')}
    </div>
    <div class="column">
        <h4 style="margin-top:0; color: var(--attention-color);">After feedback <span class="info-tooltip" data-tip="Attention recalculated after your correction was injected into the prompt.">?</span></h4>
        {extract_markdown(corrected_payload, 'attention')}
    </div>
</div>
"""

    softmax = f"""
<div class="comparison-card">
    <div class="column">
        <h4 style="margin-top:0; color: var(--softmax-color);">Before feedback <span class="info-tooltip" data-tip="Logits and probabilities the model considered before hearing the correction.">?</span></h4>
        {extract_markdown(original_payload, 'softmax')}
    </div>
    <div class="column">
        <h4 style="margin-top:0; color: var(--softmax-color);">After feedback <span class="info-tooltip" data-tip="New probability slate once the hint was added to the prompt.">?</span></h4>
        {extract_markdown(corrected_payload, 'softmax')}
    </div>
</div>
"""

    layers = f"""
<div class="comparison-card">
    <div class="column">
        <h4 style="margin-top:0; color: var(--layers-color);">Before feedback <span class="info-tooltip" data-tip="How confident each layer was in the original guess.">?</span></h4>
        {extract_markdown(original_payload, 'layers')}
    </div>
    <div class="column">
        <h4 style="margin-top:0; color: var(--layers-color);">After feedback <span class="info-tooltip" data-tip="Layer-by-layer view after the corrective hint.">?</span></h4>
        {extract_markdown(corrected_payload, 'layers')}
    </div>
</div>
"""

    delta = _render_feedback_delta(original_payload, corrected_payload)

    return summary, attention, softmax, layers, delta

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


def generate_one_word_answer(question: str, context: str = None, temperature: float = DEFAULT_TEMPERATURE) -> Tuple[str, str]:
    """
    Generate one-word answer and show layer-by-layer predictions (logit lens).

    Parameters
    ----------
    question : str
        The question to answer
    context : str, optional
        Additional context to prepend (e.g., "The answer is Green")
    temperature : float, optional
        Softmax temperature applied before sampling/logit lens visualisations

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
    result = model.generate_one_word_with_layers(
        question,
        max_new_tokens=10,
        context=context,
        temperature=temperature,
    )

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
            padding: 0.5rem 1.25rem 1.1rem 1.25rem;
            margin-bottom: 1.25rem;
            background: linear-gradient(180deg, rgba(37, 99, 235, 0.05) 0%, rgba(255, 255, 255, 0) 100%);
            border-radius: 10px;
        }

        .title-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            align-items: stretch;
            justify-content: space-between;
        }

        .title-copy {
            flex: 1 1 360px;
            min-width: 300px;
        }

        .title-heading {
            display: flex;
            align-items: baseline;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .title-main {
            font-size: 1.9rem;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: #0f172a;
            margin: 0;
        }

        .usage-label {
            font-size: 0.95rem;
            font-weight: 600;
            color: #1e3a8a;
            background: rgba(30, 58, 138, 0.12);
            border: 1px solid rgba(30, 58, 138, 0.25);
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
        }

        .title-subtitle {
            font-size: 1rem;
            color: #334155;
            margin: 0.4rem 0 0.85rem 0;
            line-height: 1.6;
            max-width: 36rem;
        }

        .usage-body {
            background: rgba(30, 64, 175, 0.05);
            border: 1px solid rgba(30, 64, 175, 0.1);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            font-size: 0.92rem;
            color: #1f2937;
            line-height: 1.65;
        }

        .usage-steps {
            margin: 0;
            padding-left: 1.2rem;
        }

        .usage-steps > li {
            margin-bottom: 0.55rem;
        }

        .usage-steps ul {
            margin-top: 0.4rem;
            padding-left: 1.1rem;
            list-style: disc;
        }

        .usage-steps ul li {
            margin-bottom: 0.25rem;
        }

        .usage-meta {
            margin-top: 0.85rem;
            font-size: 0.85rem;
            color: #475569;
            line-height: 1.6;
        }

        #pipeline-card {
            flex: 0 1 280px;
        }

        .pipeline-card {
            height: 100%;
            background: linear-gradient(180deg, rgba(14, 165, 233, 0.08) 0%, rgba(255, 255, 255, 0) 100%);
            border-radius: 12px;
            border: 1px solid rgba(14, 165, 233, 0.25);
            padding: 1.1rem 1.25rem;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.12);
        }

        .pipeline-card h4 {
            margin: 0 0 0.75rem 0;
            font-size: 1rem;
            color: #075985;
        }

        .pipeline-card ol {
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .pipeline-step {
            display: flex;
            gap: 0.6rem;
            align-items: flex-start;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(14, 165, 233, 0.18);
        }

        .pipeline-step:last-child {
            border-bottom: none;
        }

        .pipeline-check {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid rgba(14, 165, 233, 0.4);
            color: rgba(14, 165, 233, 0.9);
            flex-shrink: 0;
        }

        .pipeline-text {
            flex: 1;
        }

        .pipeline-step-label {
            display: block;
            font-weight: 600;
            font-size: 0.92rem;
            color: #0f172a;
        }

        .pipeline-text p {
            margin: 0.25rem 0 0 0;
            font-size: 0.85rem;
            color: #1f2937;
        }

        .pipeline-step-done .pipeline-check {
            background: rgba(14, 165, 233, 0.15);
        }

        .pipeline-step-active .pipeline-check {
            background: rgba(14, 165, 233, 0.08);
            color: #0284c7;
            border-color: rgba(2, 132, 199, 0.6);
        }

        .pipeline-step-pending .pipeline-check {
            background: rgba(255, 255, 255, 0.8);
            color: rgba(14, 165, 233, 0.4);
            border-style: dashed;
        }

        .pipeline-step-pending .pipeline-step-label {
            color: #1f2937;
            opacity: 0.75;
        }

        .pipeline-temperature {
            margin: 0.8rem 0 0 0;
            font-size: 0.82rem;
            color: #075985;
        }

        #temperature-note {
            font-size: 0.82rem;
            color: #1d4ed8;
            margin-top: 0.25rem;
        }

        #correction-note {
            font-size: 0.82rem;
            color: #1d4ed8;
            margin: 0.35rem 0 0.65rem 0;
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

        .info-tooltip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            background: rgba(15, 23, 42, 0.08);
            color: #1f2937;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-left: 0.3rem;
            cursor: help;
            position: relative;
        }

        .info-tooltip::after {
            content: attr(data-tip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #111827;
            color: #f8fafc;
            padding: 0.4rem 0.6rem;
            font-size: 0.75rem;
            border-radius: 4px;
            opacity: 0;
            pointer-events: none;
            white-space: nowrap;
            transition: opacity 0.15s ease;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.18);
        }

        .info-tooltip:hover::after {
            opacity: 1;
        }

        .heatmap-wrapper, .waterfall-wrapper, .sparkline-wrapper {
            margin-top: 1.25rem;
            background: rgba(15, 23, 42, 0.02);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }

        .heatmap-heading, .waterfall-heading, .sparkline-heading {
            font-size: 0.9rem;
            color: #1f2937;
            margin-bottom: 0.75rem;
            font-weight: 600;
        }

        .waterfall-note {
            margin: 0.35rem 0 0.75rem 0;
            font-size: 0.78rem;
            color: #4338ca;
        }

        .heatmap-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }

        .heatmap-table th,
        .heatmap-table td {
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 0.4rem 0.5rem;
            text-align: center;
        }

        .heatmap-table th {
            background: rgba(249, 115, 22, 0.1);
            color: #9a3412;
            font-weight: 600;
        }

        .heatmap-table th:first-child {
            text-align: left;
            min-width: 140px;
        }

        .heatmap-source {
            display: inline-block;
            margin-right: 0.45rem;
            padding: 0.1rem 0.35rem;
            border-radius: 999px;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #7c2d12;
            background: rgba(249, 115, 22, 0.15);
        }

        .heatmap-source-question,
        .heatmap-source-prompt {
            color: #1e3a8a;
            background: rgba(30, 58, 138, 0.1);
        }

        .heatmap-source-context {
            color: #166534;
            background: rgba(22, 101, 52, 0.12);
        }

        .heatmap-value {
            font-weight: 500;
            color: #7c2d12;
        }

        .waterfall-row {
            display: grid;
            grid-template-columns: 1.4fr 1fr 1.8fr 1.2fr;
            gap: 0.75rem;
            align-items: center;
            padding: 0.55rem 0.35rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }

        .waterfall-row:last-child {
            border-bottom: none;
        }

        .waterfall-token {
            font-weight: 600;
            color: #312e81;
        }

        .waterfall-stage {
            font-size: 0.78rem;
            color: #334155;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .waterfall-bar {
            position: relative;
            height: 8px;
            border-radius: 999px;
            flex: 0 0 90px;
        }

        .waterfall-bar.exp {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.15), rgba(99, 102, 241, 0.6));
        }

        .waterfall-bar.prob {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.2), rgba(99, 102, 241, 0.8));
        }

        .sparkline {
            width: 100%;
            height: auto;
        }

        .sparkline-area {
            fill: rgba(14, 165, 233, 0.15);
        }

        .sparkline-line {
            fill: none;
            stroke: rgba(14, 165, 233, 0.9);
            stroke-width: 2;
        }

        .sparkline-axis-line {
            stroke: #94a3b8;
            stroke-width: 1;
        }

        .sparkline-tick-mark {
            stroke: #cbd5f5;
            stroke-width: 1;
        }

        .sparkline-tick-label {
            font-size: 0.65rem;
            fill: #475569;
            text-anchor: middle;
            dominant-baseline: hanging;
        }

        .sparkline-axis-caption {
            font-size: 0.7rem;
            fill: #475569;
            text-anchor: end;
            font-weight: 600;
        }

        .delta-wrapper {
            margin-top: 1.5rem;
            background: rgba(15, 23, 42, 0.02);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
        }

        .delta-heading {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.9rem;
        }

        .delta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }

        .delta-card {
            background: #fff;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 0.9rem 1rem;
            box-shadow: 0 4px 6px rgba(15, 23, 42, 0.04);
        }

        .delta-card h4 {
            margin: 0 0 0.6rem 0;
            font-size: 0.9rem;
            color: #1f2937;
        }

        .delta-card ul {
            margin: 0;
            padding-left: 1rem;
            color: #334155;
            font-size: 0.8rem;
        }

        .delta-label {
            font-weight: 600;
            color: #111827;
        }

        .delta-value {
            margin-left: 0.4rem;
            font-family: "JetBrains Mono", "Menlo", monospace;
            color: #1d4ed8;
        }
        """
    ) as demo:

        with gr.Column(elem_id="title-section"):
            with gr.Row(elem_id="title-grid"):
                gr.HTML(
                    """
                    <div class="title-copy">
                        <div class="title-heading">
                            <h1 class="title-main">LLM under the hood</h1>
                        </div>
                        <p class="title-subtitle">Peek inside a transformer to follow how attention, layers, and softmax work together to choose the next word.</p>
                        <div class="usage-body">
                            <span class="usage-label">How to use this app</span>
                            <ol class="usage-steps">
                                <li>Ask a crisp factual question — single-word answers make the internals easiest to read.</li>
                                <li>Adjust the <strong>Temperature</strong> slider (right under the question box) to compare steady vs exploratory behaviour before you run.</li>
                                <li>Scan the Query tabs from left to right to connect the attention bars, heatmap, softmax waterfall, and confidence sparkline to the answer.</li>
                                <li>After the first try choose one feedback path:
                                    <ul>
                                        <li><strong>That's Wrong</strong> tells TinyLlama to retry while keeping your question the same.</li>
                                        <li><strong>Submit Correction</strong> sends the exact word you expect so the next turn can aim directly at it.</li>
                                    </ul>
                                </li>
                            </ol>
                        </div>
                        <p class="usage-meta"><strong>Model:</strong> TinyLlama-1.1B is an open 1.1&nbsp;billion parameter chat transformer. It's far smaller than ChatGPT’s GPT-4, so it may make mistakes, but the compact size lets us expose every attention head, logit, and layer.</p>
                    </div>
                    """,
                    elem_id="title-copy",
                )
                pipeline_progress = gr.HTML(
                    value=_build_pipeline_html(0, None, DEFAULT_TEMPERATURE),
                    visible=True,
                    elem_id="pipeline-card",
                )

        gr.HTML('<div class="section-divider"></div>')

        empty_attention_html = "<em>Run the model to list attention percentages for every word in your prompt and context.</em>"
        empty_attention_heatmap = "<em>Run the model to visualise the full attention heatmap.</em>"
        empty_softmax_html = "<em>Run the model to reveal the top candidate words and their probabilities.</em>"
        empty_softmax_waterfall = "<em>Run the model to watch logits turn into probabilities.</em>"
        empty_layers_html = "<em>Run the model to chart how confidence evolves across layers.</em>"
        empty_layers_sparkline = "<em>Run the model to trace the confidence sparkline.</em>"
        correction_placeholder = "_Generate an answer and press “That’s Wrong” to compare before and after._"

        with gr.Tabs():
            with gr.Tab("Query"):
                gr.Markdown("### Ask the model a quick fact")
                gr.Markdown(
                    "Use this tab to follow the model's first attempt. Short prompts such as **What is the capital of Australia?** work best because the answer should be a single word. Want the full story? 📘 [Read the overview theory](#theory-overview)."
                )

                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is the capital of Australia?",
                    lines=1,
                    show_label=True,
                    container=False
                )
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1.5,
                    step=0.05,
                    value=DEFAULT_TEMPERATURE,
                    info="Lower = confident & deterministic · Higher = exploratory & diverse",
                    interactive=True,
                )

                gr.Markdown(
                    "💡 **Temperature tip:** This slider rescales the logits before softmax. Cool settings (≈0.1–0.3) make the top token dominate, while warmer settings (≈0.8–1.2) keep more candidates alive so you can see probabilities redistribute.",
                    elem_id="temperature-note",
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
                            "Every word from your prompt and any added context appears here with an attention percentage so you can see what the model reread. 📘 [Open the attention theory](#theory-attention)."
                        )
                        query_attention_panel = gr.HTML(empty_attention_html)
                        query_attention_heatmap = gr.HTML(empty_attention_heatmap)

                    with gr.Tab("🟣 Softmax"):
                        gr.Markdown(
                            "Purple bars reveal the probability of the highest-scoring candidate words just before the model spoke. 📘 [Open the softmax theory](#theory-softmax)."
                        )
                        query_softmax_panel = gr.HTML(empty_softmax_html)
                        query_softmax_waterfall = gr.HTML(empty_softmax_waterfall)

                    with gr.Tab("🔵 Layer by Layer"):
                        gr.Markdown(
                            "Blue cards track how confidence in the final word grows as information flows through deeper transformer layers. 📘 [Open the layer theory](#theory-layers)."
                        )
                        query_layers_panel = gr.HTML(empty_layers_html)
                        query_layers_sparkline = gr.HTML(empty_layers_sparkline)

            with gr.Tab("Correction"):
                gr.Markdown("### Teach the model when it slips")
                gr.Markdown(
                    "Challenge the answer and optionally provide the correct word. The comparison panels show exactly how the internal signals respond to your feedback. 📘 [See why feedback matters](#theory-feedback)."
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

                gr.Markdown(
                    "_Choose one path per turn: use **That's Wrong** when you just want the model to try again with a gentle nudge, or provide the exact word and press **Submit Correction** to steer it directly._",
                    elem_id="correction-note",
                )

                correction_summary_md = gr.Markdown(correction_placeholder)

                with gr.Tabs():
                    with gr.Tab("🟠 Attention"):
                        gr.Markdown("See how focus on each prompt or context word changes once the correction is in play. 📘 [Revisit the attention theory](#theory-attention).")
                        correction_attention_panel = gr.HTML("<em>No comparison yet.</em>")

                    with gr.Tab("🟣 Softmax"):
                        gr.Markdown("Check whether the corrected word overtakes its competitors in the probability slate. 📘 [Revisit the softmax theory](#theory-softmax).")
                        correction_softmax_panel = gr.HTML("<em>No comparison yet.</em>")

                    with gr.Tab("🔵 Layer by Layer"):
                        gr.Markdown("Look for deeper layers to embrace the corrected answer sooner or with greater certainty. 📘 [Revisit the layer theory](#theory-layers).")
                        correction_layers_panel = gr.HTML("<em>No comparison yet.</em>")

                    with gr.Tab("Δ Impact"):
                        gr.Markdown("Read this tab for the biggest attention, probability, and layer shifts triggered by your feedback. 📘 [Review the feedback theory](#theory-feedback).")
                        correction_delta_panel = gr.HTML("<em>No comparison yet.</em>")

            with gr.Tab("Theory"):
                gr.Markdown("### Peek behind the math")
                gr.Markdown(
                    "These notes explain the coloured panels in everyday language before linking them back to the underlying equations."
                )

                theory_overview_md = gr.Markdown(THEORY_TEXT["overview"], elem_id="theory-overview")
                theory_feedback_md = gr.Markdown(THEORY_TEXT["feedback"], elem_id="theory-feedback")

                with gr.Tabs():
                    with gr.Tab("🟠 Attention"):
                        theory_attention_md = gr.Markdown(THEORY_TEXT["attention"], elem_id="theory-attention")

                    with gr.Tab("🟣 Softmax"):
                        theory_softmax_md = gr.Markdown(THEORY_TEXT["softmax"], elem_id="theory-softmax")

                    with gr.Tab("🔵 Layer by Layer"):
                        theory_layers_md = gr.Markdown(THEORY_TEXT["layers"], elem_id="theory-layers")

        # Event handlers
        # Store original answer to avoid re-generation
        original_answer_cache: Dict[Tuple[str, float], Dict[str, object]] = {}

        def _make_cache_key(question: str, temperature: float) -> Tuple[str, float]:
            return (question.strip(), round(float(temperature), 3))

        def _build_idle_pipeline(temperature: float):
            return gr.update(
                value=_build_pipeline_html(0, None, temperature),
                visible=True,
            )

        def _pipeline_prepare(question: str, temperature: float):
            stripped = (question or "").strip()
            if not stripped:
                return _build_idle_pipeline(temperature)
            return gr.update(
                value=_build_pipeline_html(completed_steps=1, active_step=2, temperature=temperature),
                visible=True,
            )

        def _pipeline_prepare_with_correction(question: str, _correction: str, temperature: float):
            return _pipeline_prepare(question, temperature)

        def on_temperature_change(temperature: float):
            return _build_idle_pipeline(temperature)

        def on_generate(question, temperature):
            """Generate answer and update all tabs."""
            if not question.strip():
                return (
                    "Please enter a question!",
                    "_Ask something like “What is the capital of Australia?”_",
                    _build_idle_pipeline(temperature),
                    empty_attention_html,
                    empty_attention_html,
                    empty_attention_heatmap,
                    empty_softmax_html,
                    empty_softmax_waterfall,
                    empty_layers_html,
                    empty_layers_sparkline,
                    THEORY_TEXT["overview"],
                    THEORY_TEXT["feedback"],
                    THEORY_TEXT["attention"],
                    THEORY_TEXT["softmax"],
                    THEORY_TEXT["layers"],
                    correction_placeholder,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            answer, payload = generate_one_word_answer(question, temperature=temperature)

            if isinstance(payload, str):
                # Error or guidance message
                pipeline_update = _build_idle_pipeline(temperature)
                return (
                    answer,
                    payload,
                    pipeline_update,
                    empty_attention_html,
                    empty_attention_heatmap,
                    empty_softmax_html,
                    empty_softmax_waterfall,
                    empty_layers_html,
                    empty_layers_sparkline,
                    THEORY_TEXT["overview"],
                    THEORY_TEXT["feedback"],
                    THEORY_TEXT["attention"],
                    THEORY_TEXT["softmax"],
                    THEORY_TEXT["layers"],
                    correction_placeholder,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            pipeline_update = gr.update(
                value=_build_pipeline_html(len(PIPELINE_STEPS), None, temperature),
                visible=True,
            )

            cache_key = _make_cache_key(question, temperature)
            original_answer_cache[cache_key] = {
                "answer": answer,
                "payload": payload,
            }

            return (
                answer,
                payload["overview"],
                pipeline_update,
                payload["attention"]["markdown"],
                payload["attention"]["heatmap"],
                payload["softmax"]["markdown"],
                payload["softmax"]["waterfall"],
                payload["layers"]["markdown"],
                payload["layers"]["sparkline"],
                payload["theory"]["overview"],
                payload["theory"]["feedback"],
                payload["theory"]["attention"],
                payload["theory"]["softmax"],
                payload["theory"]["layers"],
                "_Use “That’s Wrong” for a quick retry or add the exact word and press “Submit Correction.”_",
                "<em>No comparison yet.</em>",
                "<em>No comparison yet.</em>",
                "<em>No comparison yet.</em>",
                "<em>No comparison yet.</em>",
            )

        def on_wrong_clicked(question, temperature):
            """Handle 'That's Wrong!' button - tell model its answer was wrong."""
            stripped_question = (question or "").strip()
            cache_key = _make_cache_key(stripped_question, temperature)
            entry = original_answer_cache.get(cache_key)
            if entry is None and stripped_question:
                entry = next(
                    (
                        payload
                        for (stored_question, _), payload in original_answer_cache.items()
                        if stored_question == stripped_question
                    ),
                    None,
                )
            if not stripped_question or entry is None:
                return (
                    _build_idle_pipeline(temperature),
                    "⚠️ Please generate an answer first!",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            original_entry = entry
            original_answer = original_entry["answer"]
            original_payload = original_entry["payload"]

            correction_context = f"{original_answer} is wrong, the right answer is"
            corrected_answer, corrected_payload = generate_one_word_answer(
                question,
                context=correction_context,
                temperature=temperature,
            )

            if isinstance(corrected_payload, str):
                return (
                    _build_idle_pipeline(temperature),
                    corrected_payload,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            summary, attention_html, softmax_html, layers_html, delta_html = build_correction_sections(
                question,
                original_answer,
                original_payload,
                correction_context,
                corrected_answer,
                corrected_payload,
            )

            pipeline_update = gr.update(
                value=_build_pipeline_html(len(PIPELINE_STEPS), None, temperature),
                visible=True,
            )

            return pipeline_update, summary, attention_html, softmax_html, layers_html, delta_html

        def on_correction(question, correction, temperature):
            """Handle manual correction with specific answer."""
            if not correction.strip():
                return (
                    _build_idle_pipeline(temperature),
                    "⚠️ Please enter a correction!",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            stripped_question = (question or "").strip()
            cache_key = _make_cache_key(stripped_question, temperature)
            entry = original_answer_cache.get(cache_key)
            if entry is None and stripped_question:
                entry = next(
                    (
                        payload
                        for (stored_question, _), payload in original_answer_cache.items()
                        if stored_question == stripped_question
                    ),
                    None,
                )
            if not stripped_question or entry is None:
                return (
                    _build_idle_pipeline(temperature),
                    "⚠️ Please generate an answer first!",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            original_entry = entry
            original_answer = original_entry["answer"]
            original_payload = original_entry["payload"]

            correction_context = f"{original_answer} is wrong, the right answer is {correction.strip()}"
            corrected_answer, corrected_payload = generate_one_word_answer(
                question,
                context=correction_context,
                temperature=temperature,
            )

            if isinstance(corrected_payload, str):
                return (
                    _build_idle_pipeline(temperature),
                    corrected_payload,
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                    "<em>No comparison yet.</em>",
                )

            summary, attention_html, softmax_html, layers_html, delta_html = build_correction_sections(
                question,
                original_answer,
                original_payload,
                correction_context,
                corrected_answer,
                corrected_payload,
            )

            pipeline_update = gr.update(
                value=_build_pipeline_html(len(PIPELINE_STEPS), None, temperature),
                visible=True,
            )

            return pipeline_update, summary, attention_html, softmax_html, layers_html, delta_html

        generate_btn.click(
            fn=_pipeline_prepare,
            inputs=[question_input, temperature_slider],
            outputs=[pipeline_progress],
            queue=False,
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[question_input, temperature_slider],
            outputs=[
                response_output,
                query_overview_md,
                pipeline_progress,
                query_attention_panel,
                query_attention_heatmap,
                query_softmax_panel,
                query_softmax_waterfall,
                query_layers_panel,
                query_layers_sparkline,
                theory_overview_md,
                theory_feedback_md,
                theory_attention_md,
                theory_softmax_md,
                theory_layers_md,
                correction_summary_md,
                correction_attention_panel,
                correction_softmax_panel,
                correction_layers_panel,
                correction_delta_panel,
            ],
            show_progress=False
        )

        wrong_btn.click(
            fn=_pipeline_prepare,
            inputs=[question_input, temperature_slider],
            outputs=[pipeline_progress],
            queue=False,
        )

        wrong_btn.click(
            fn=on_wrong_clicked,
            inputs=[question_input, temperature_slider],
            outputs=[
                pipeline_progress,
                correction_summary_md,
                correction_attention_panel,
                correction_softmax_panel,
                correction_layers_panel,
                correction_delta_panel,
            ],
            show_progress=False
        )

        correction_btn.click(
            fn=_pipeline_prepare_with_correction,
            inputs=[question_input, correction_input, temperature_slider],
            outputs=[pipeline_progress],
            queue=False,
        )

        correction_btn.click(
            fn=on_correction,
            inputs=[question_input, correction_input, temperature_slider],
            outputs=[
                pipeline_progress,
                correction_summary_md,
                correction_attention_panel,
                correction_softmax_panel,
                correction_layers_panel,
                correction_delta_panel,
            ],
            show_progress=False
        )

        temperature_slider.change(
            fn=on_temperature_change,
            inputs=[temperature_slider],
            outputs=[pipeline_progress],
            queue=False,
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
