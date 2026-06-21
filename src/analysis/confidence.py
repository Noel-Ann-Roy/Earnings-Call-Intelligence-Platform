"""
src/analysis/confidence.py
==========================
Phase 6: Management Confidence Score

METHODOLOGY:
  The confidence score synthesizes all NLP signals into a single
  0-100 metric that reflects overall management tone and conviction.

FORMULA:
  Score = (Sentiment × 0.40) + (Growth × 0.30) + (Guidance × 0.20) - (Risk × 0.10)

WHY THESE WEIGHTS?
  - Sentiment (40%): The overall emotional tone of the ENTIRE transcript
    is the strongest predictor of management confidence.
  - Growth (30%): Concrete growth signals show the business is performing.
    Management tends to be confident when fundamentals are strong.
  - Guidance (20%): Issuing specific guidance requires confidence.
    Vague or no guidance signals uncertainty.
  - Risk (-10%): Risk mentions negatively impact confidence.
    Note the small weight: mentioning risks doesn't mean management
    is not confident — disclosing risks is required by SEC rules.

IMPORTANT:
  This score measures COMMUNICATION TONE, not financial health.
  A company can have a high confidence score but poor fundamentals,
  or vice versa.

VERDICT LOGIC:
  Score ≥ 65 → BULLISH   (Tone is strongly positive and specific)
  Score 40–64 → NEUTRAL  (Mixed or hedged tone)
  Score < 40  → BEARISH  (Cautious, vague, or risk-heavy tone)
"""

from typing import Dict


# Component weights — must sum to 1.0
# (Risk weight is subtracted, so net = 0.40 + 0.30 + 0.20 - 0.10 = 0.80 + deduction)
WEIGHTS = {
    "sentiment": 0.40,
    "growth": 0.30,
    "guidance": 0.20,
    "risk_deduction": 0.10,
}

VERDICT_THRESHOLDS = {
    "BULLISH": 65,
    "NEUTRAL": 40,
    # Below 40 → BEARISH
}

VERDICT_COLORS = {
    "BULLISH": "#00C805",   # Green
    "NEUTRAL": "#FFA500",   # Orange
    "BEARISH": "#FF3B3B",   # Red
}

VERDICT_ICONS = {
    "BULLISH": "📈",
    "NEUTRAL": "📊",
    "BEARISH": "📉",
}


def compute_confidence_score(
    sentiment_result: Dict,
    growth_result: Dict,
    guidance_result: Dict,
    risk_result: Dict,
) -> Dict:
    """
    Compute the Management Confidence Score from all NLP module outputs.

    Args:
        sentiment_result: Output from FinBERTAnalyzer.analyze()
        growth_result:    Output from detect_growth_signals()
        guidance_result:  Output from extract_guidance()
        risk_result:      Output from detect_risks()

    Returns:
        Dict containing:
          - score: float (0-100)
          - components: Dict showing each component's contribution
          - verdict: str ("BULLISH" / "NEUTRAL" / "BEARISH")
          - verdict_color: str (hex color for UI)
          - reasoning: str (plain-English explanation)
          - formula_display: str (formatted formula for display)
    """

    # ── Component 1: Sentiment (0-100) ──────────────────────────────────────
    # FinBERT positive_score is a probability (0-1)
    # We scale it: 0.5 positive → 50/100, 0.9 positive → 90/100
    # We also penalize if negative_score is high
    pos = sentiment_result.get("positive_score", 0.33)
    neg = sentiment_result.get("negative_score", 0.33)
    # Net positive ratio, scaled 0-100
    sentiment_component = max(0, (pos - neg + 1) / 2) * 100

    # ── Component 2: Growth Score (0-100) ───────────────────────────────────
    growth_component = float(growth_result.get("growth_score", 0))

    # ── Component 3: Guidance Score (0-100) ─────────────────────────────────
    guidance_component = float(guidance_result.get("guidance_score", 0))

    # ── Component 4: Risk Deduction (0-100) ─────────────────────────────────
    risk_component = float(risk_result.get("risk_score", 0))

    # ── Final Formula ─────────────────────────────────────────────────────────
    raw_score = (
        sentiment_component * WEIGHTS["sentiment"]
        + growth_component * WEIGHTS["growth"]
        + guidance_component * WEIGHTS["guidance"]
        - risk_component * WEIGHTS["risk_deduction"]
    )

    # Clamp to [0, 100]
    score = max(0.0, min(100.0, raw_score))

    # ── Verdict ───────────────────────────────────────────────────────────────
    if score >= VERDICT_THRESHOLDS["BULLISH"]:
        verdict = "BULLISH"
    elif score >= VERDICT_THRESHOLDS["NEUTRAL"]:
        verdict = "NEUTRAL"
    else:
        verdict = "BEARISH"

    # ── Reasoning ─────────────────────────────────────────────────────────────
    reasoning = _generate_reasoning(
        score, verdict, sentiment_component, growth_component,
        guidance_component, risk_component,
        sentiment_result, growth_result, guidance_result, risk_result,
    )

    # ── Formula Display ───────────────────────────────────────────────────────
    formula_display = (
        f"Score = ({sentiment_component:.1f} × 0.40) "
        f"+ ({growth_component:.1f} × 0.30) "
        f"+ ({guidance_component:.1f} × 0.20) "
        f"- ({risk_component:.1f} × 0.10) "
        f"= **{score:.1f}**"
    )

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "verdict_color": VERDICT_COLORS[verdict],
        "verdict_icon": VERDICT_ICONS[verdict],
        "components": {
            "sentiment": {
                "value": round(sentiment_component, 1),
                "weight": WEIGHTS["sentiment"],
                "contribution": round(sentiment_component * WEIGHTS["sentiment"], 1),
                "label": "Sentiment",
            },
            "growth": {
                "value": round(growth_component, 1),
                "weight": WEIGHTS["growth"],
                "contribution": round(growth_component * WEIGHTS["growth"], 1),
                "label": "Growth Signals",
            },
            "guidance": {
                "value": round(guidance_component, 1),
                "weight": WEIGHTS["guidance"],
                "contribution": round(guidance_component * WEIGHTS["guidance"], 1),
                "label": "Guidance",
            },
            "risk": {
                "value": round(risk_component, 1),
                "weight": WEIGHTS["risk_deduction"],
                "contribution": round(-risk_component * WEIGHTS["risk_deduction"], 1),
                "label": "Risk Deduction",
            },
        },
        "reasoning": reasoning,
        "formula_display": formula_display,
    }


def _generate_reasoning(
    score, verdict,
    sentiment_c, growth_c, guidance_c, risk_c,
    sentiment_r, growth_r, guidance_r, risk_r
) -> str:
    """Generate a plain-English explanation of the confidence score."""

    lines = []

    # Sentiment commentary
    sent_label = sentiment_r.get("overall_sentiment", "Neutral")
    sent_conf = sentiment_r.get("confidence", 50)
    lines.append(
        f"**Sentiment**: Management tone is {sent_label} with "
        f"{sent_conf:.0f}% confidence (score: {sentiment_c:.0f}/100)."
    )

    # Growth commentary
    n_signals = growth_r.get("categories_detected", 0)
    strong = growth_r.get("strong_signals", 0)
    lines.append(
        f"**Growth**: {n_signals} growth signal categories detected, "
        f"{strong} of which are strong signals (score: {growth_c:.0f}/100)."
    )

    # Guidance commentary
    n_guidance = guidance_r.get("total_statements", 0)
    tone_counts = guidance_r.get("tone_counts", {})
    optimistic = tone_counts.get("Optimistic", 0)
    lines.append(
        f"**Guidance**: {n_guidance} forward-looking statements found, "
        f"{optimistic} with an optimistic tone (score: {guidance_c:.0f}/100)."
    )

    # Risk commentary
    n_risks = risk_r.get("risk_categories_detected", 0)
    lines.append(
        f"**Risk**: {n_risks} risk categories flagged "
        f"(deduction score: {risk_c:.0f}/100)."
    )

    # Verdict summary
    if verdict == "BULLISH":
        lines.append(
            f"\n**Verdict — BULLISH ({score:.0f}/100)**: Management demonstrates "
            f"strong confidence with an optimistic tone, multiple growth drivers, "
            f"and clear forward guidance."
        )
    elif verdict == "NEUTRAL":
        lines.append(
            f"\n**Verdict — NEUTRAL ({score:.0f}/100)**: Management shows mixed signals. "
            f"Positive developments are present but balanced by cautious language "
            f"or elevated risk mentions."
        )
    else:
        lines.append(
            f"\n**Verdict — BEARISH ({score:.0f}/100)**: Management tone is cautious. "
            f"Risk mentions dominate, guidance is limited, and growth signals are weak."
        )

    return "\n\n".join(lines)
