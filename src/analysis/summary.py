"""
src/analysis/summary.py
=======================
Executive Summary Generator

Generates an analyst-style narrative summary by combining all NLP outputs.

APPROACH:
  Template-based generation with dynamic insertion of NLP results.
  This is more reliable and explainable than LLM-generated summaries
  for structured financial analysis.

  The summary covers:
    1. Overall Management Tone (from FinBERT)
    2. Major Growth Drivers (from growth signals)
    3. Key Risks (from risk detection)
    4. Forward Guidance Outlook (from guidance extraction)
    5. Analyst Assessment (from confidence score)
"""

from typing import Dict, List


def generate_executive_summary(
    sentiment_result: Dict,
    growth_result: Dict,
    risk_result: Dict,
    guidance_result: Dict,
    confidence_result: Dict,
    company_name: str = "Management",
) -> str:
    """
    Generate a structured analyst-style executive summary.

    Args:
        sentiment_result:  FinBERT output
        growth_result:     Growth signal output
        risk_result:       Risk detection output
        guidance_result:   Guidance extraction output
        confidence_result: Confidence score output
        company_name:      Company name for the report header

    Returns:
        Formatted markdown string for display in Streamlit
    """
    lines = []

    # ── Header ─────────────────────────────────────────────────────────────
    verdict = confidence_result.get("verdict", "NEUTRAL")
    score = confidence_result.get("score", 50)
    lines.append(f"## Executive Summary")
    lines.append(
        f"**Overall Assessment**: {verdict} | "
        f"**Management Confidence Score**: {score}/100"
    )
    lines.append("")

    # ── 1. Management Tone ─────────────────────────────────────────────────
    lines.append("### 1. Management Tone")
    sent = sentiment_result.get("overall_sentiment", "Neutral")
    pos = sentiment_result.get("positive_score", 0)
    neg = sentiment_result.get("negative_score", 0)
    neu = sentiment_result.get("neutral_score", 0)
    conf = sentiment_result.get("confidence", 50)

    tone_desc = {
        "Positive": "Management maintained an optimistic and confident tone throughout the call.",
        "Neutral": "Management delivered primarily factual, data-driven commentary with limited emotional signaling.",
        "Negative": "Management displayed cautious or defensive language, suggesting uncertainty about near-term performance.",
    }

    lines.append(tone_desc.get(sent, "Management tone was mixed."))
    lines.append(
        f"FinBERT sentiment analysis across all transcript sentences yielded: "
        f"Positive {pos:.1%}, Neutral {neu:.1%}, Negative {neg:.1%} "
        f"(confidence: {conf:.0f}%)."
    )
    lines.append("")

    # ── 2. Major Growth Drivers ─────────────────────────────────────────────
    lines.append("### 2. Major Growth Drivers")
    signals = growth_result.get("signals", [])
    if signals:
        strong = [s for s in signals if s["strength"] == "Strong"]
        moderate = [s for s in signals if s["strength"] == "Moderate"]

        if strong:
            lines.append(
                f"**Strong signals** ({len(strong)} categories): "
                + ", ".join(s["category"] for s in strong) + "."
            )
        if moderate:
            lines.append(
                f"**Moderate signals** ({len(moderate)} categories): "
                + ", ".join(s["category"] for s in moderate) + "."
            )

        # Top signal detail
        top = signals[0]
        if top.get("example_sentence"):
            lines.append(
                f"The most prominent growth driver is **{top['category']}**, "
                f"mentioned {top['count']} times. Example: "
                f'*"{top["example_sentence"][:200]}"*'
            )
    else:
        lines.append(
            "No significant growth signals were detected. "
            "Management focused on operational updates without highlighting growth drivers."
        )
    lines.append("")

    # ── 3. Key Risks ────────────────────────────────────────────────────────
    lines.append("### 3. Key Risks")
    risks = risk_result.get("risks", [])
    high_risks = [r for r in risks if r["severity"] == "High"]
    med_risks = [r for r in risks if r["severity"] == "Medium"]

    if high_risks:
        lines.append(
            f"**High-severity risks** ({len(high_risks)}): "
            + ", ".join(r["category"] for r in high_risks) + "."
        )
    if med_risks:
        lines.append(
            f"**Medium-severity risks** ({len(med_risks)}): "
            + ", ".join(r["category"] for r in med_risks) + "."
        )
    if not risks:
        lines.append(
            "No significant risk signals detected. "
            "Management did not emphasize risk factors in this call."
        )
    else:
        total = risk_result.get("total_risk_mentions", 0)
        lines.append(f"Total risk mentions across the transcript: {total}.")
    lines.append("")

    # ── 4. Future Outlook / Guidance ────────────────────────────────────────
    lines.append("### 4. Future Outlook & Guidance")
    guidance_stmts = guidance_result.get("statements", [])
    tone_counts = guidance_result.get("tone_counts", {})

    if guidance_stmts:
        n = len(guidance_stmts)
        opt = tone_counts.get("Optimistic", 0)
        caut = tone_counts.get("Cautious", 0)
        lines.append(
            f"Management issued {n} forward-looking statements. "
            f"Of these, {opt} were optimistic and {caut} were cautious in tone."
        )

        # Show top guidance by type
        by_type = guidance_result.get("by_type", {})
        if "Revenue Guidance" in by_type:
            rev = by_type["Revenue Guidance"][0]["sentence"]
            lines.append(f'**Revenue Guidance**: *"{rev[:250]}"*')
        if "Margin Guidance" in by_type:
            mar = by_type["Margin Guidance"][0]["sentence"]
            lines.append(f'**Margin Guidance**: *"{mar[:250]}"*')
    else:
        lines.append(
            "No explicit forward guidance was detected. "
            "This may indicate that management withdrew or did not provide formal guidance."
        )
    lines.append("")

    # ── 5. Final Assessment ─────────────────────────────────────────────────
    lines.append("### 5. Final Assessment")
    lines.append(confidence_result.get("reasoning", "No reasoning available."))

    return "\n".join(lines)
