"""
src/analysis/guidance.py
========================
Forward-Looking Statement Extraction

Guidance statements are forward-looking: they tell investors what management
EXPECTS to happen in future quarters. Detecting them is crucial because:
  - Strong, specific guidance → management confidence
  - Vague or withdrawn guidance → uncertainty

APPROACH:
  We use a two-step rule-based pipeline:
  1. Detect "trigger phrases" that mark forward-looking statements
     (e.g., "we expect", "we anticipate", "we forecast")
  2. Classify the guidance type by scanning for category keywords
     (Revenue, Margin, EPS, Growth, Capex, etc.)

WHY NOT ML?
  SEC regulations (Safe Harbor provisions) define forward-looking statements
  very precisely. The trigger phrases are standardized. Rule-based extraction
  matches human analyst accuracy here without needing labeled data.

REGULATORY NOTE:
  Forward-looking statements are disclosed under SEC Safe Harbor rules.
  Companies always hedge them: "subject to risks and uncertainties."
  This is normal — not a negative signal.
"""

import re
from typing import Dict, List

# Phrases that signal a forward-looking statement
GUIDANCE_TRIGGERS = [
    "we expect",
    "we anticipate",
    "we forecast",
    "we project",
    "we guide",
    "we are guiding",
    "we believe",
    "we see",
    "we target",
    "outlook is",
    "guidance is",
    "full year guidance",
    "next quarter",
    "next fiscal",
    "going forward",
    "for the year",
    "fiscal year",
    "q1 guidance",
    "q2 guidance",
    "q3 guidance",
    "q4 guidance",
    "annual guidance",
    "full-year",
    "second half",
    "first half",
]

# Guidance type classification
GUIDANCE_TYPES = {
    "Revenue Guidance": [
        "revenue", "sales", "top-line", "net revenue", "total revenue",
    ],
    "EPS / Earnings Guidance": [
        "eps", "earnings per share", "diluted eps", "net income",
        "earnings", "profit",
    ],
    "Margin Guidance": [
        "gross margin", "operating margin", "ebitda margin", "margin",
    ],
    "Growth Guidance": [
        "growth", "grow", "increase", "expand",
    ],
    "Capex Guidance": [
        "capital expenditure", "capex", "investment", "r&d",
    ],
    "Headcount / Hiring": [
        "headcount", "hiring", "workforce", "employees",
    ],
    "General Outlook": [],  # Fallback if no specific type matched
}

# Confidence qualifiers
POSITIVE_QUALIFIERS = [
    "strong", "robust", "solid", "healthy", "accelerate", "exceed",
    "above", "outperform", "beat",
]
CAUTIOUS_QUALIFIERS = [
    "challenging", "headwind", "uncertain", "difficult", "below",
    "pressure", "slower",
]


def _classify_guidance_type(sentence: str) -> str:
    """Classify a guidance sentence into a guidance type."""
    sentence_lower = sentence.lower()
    for guidance_type, keywords in GUIDANCE_TYPES.items():
        if guidance_type == "General Outlook":
            continue
        if any(kw in sentence_lower for kw in keywords):
            return guidance_type
    return "General Outlook"


def _assess_tone(sentence: str) -> str:
    """Assess whether the guidance is optimistic, cautious, or neutral."""
    sentence_lower = sentence.lower()
    positive_hits = sum(1 for q in POSITIVE_QUALIFIERS if q in sentence_lower)
    cautious_hits = sum(1 for q in CAUTIOUS_QUALIFIERS if q in sentence_lower)

    if positive_hits > cautious_hits:
        return "Optimistic"
    elif cautious_hits > positive_hits:
        return "Cautious"
    else:
        return "Neutral"


def extract_guidance(sentences: List[str]) -> Dict:
    """
    Extract forward-looking guidance statements from transcript sentences.

    Args:
        sentences: List of transcript sentences

    Returns:
        Dict containing:
          - statements: List of guidance statement dicts
          - guidance_score: float (0-100, used in confidence formula)
          - by_type: Dict grouping statements by guidance type
    """
    statements = []

    for sentence in sentences:
        sentence_lower = sentence.lower()

        # Check if sentence contains a guidance trigger
        triggered = any(trigger in sentence_lower for trigger in GUIDANCE_TRIGGERS)
        if not triggered:
            continue

        # Skip very short sentences (likely false positives)
        if len(sentence.split()) < 8:
            continue

        guidance_type = _classify_guidance_type(sentence)
        tone = _assess_tone(sentence)

        statements.append({
            "sentence": sentence.strip(),
            "type": guidance_type,
            "tone": tone,
        })

    # Deduplicate near-identical statements
    seen = set()
    unique_statements = []
    for stmt in statements:
        key = stmt["sentence"][:80].lower()
        if key not in seen:
            seen.add(key)
            unique_statements.append(stmt)

    # Group by type
    by_type: Dict[str, List] = {}
    for stmt in unique_statements:
        gtype = stmt["type"]
        by_type.setdefault(gtype, []).append(stmt)

    # Count tone distribution
    tone_counts = {"Optimistic": 0, "Cautious": 0, "Neutral": 0}
    for stmt in unique_statements:
        tone_counts[stmt["tone"]] += 1

    # Guidance score: more statements + more optimistic → higher score
    n = len(unique_statements)
    optimism_ratio = tone_counts["Optimistic"] / max(n, 1)
    guidance_score = min(100, (n * 8) + (optimism_ratio * 40))

    return {
        "statements": unique_statements,
        "by_type": by_type,
        "total_statements": n,
        "tone_counts": tone_counts,
        "guidance_score": round(guidance_score, 1),
    }
