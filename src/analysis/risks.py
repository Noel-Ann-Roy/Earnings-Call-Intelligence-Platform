"""
src/analysis/risks.py
=====================
Phase 5a: Rule-Based Risk Detection

APPROACH:
  We use keyword matching + context extraction instead of ML here.

WHY RULE-BASED FOR RISK?
  - Risk categories are well-defined and domain-specific
  - Keyword matching gives high precision for known risk types
  - No labeled data needed
  - Fully interpretable: you can see exactly WHY a sentence was flagged
  - Works out of the box with no training or fine-tuning

  ML models for risk classification would require thousands of labeled
  earnings call sentences, which is expensive to create. Rule-based
  extraction is the industry standard for structured information extraction
  from financial documents (used by Bloomberg, FactSet, etc.)

SEVERITY LOGIC:
  High:   3+ keyword matches in one sentence, or critical terms
  Medium: 2 keyword matches
  Low:    1 keyword match
"""

import re
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────
# RISK TAXONOMY
# Each risk category has:
#   - keywords: terms that signal this risk (case-insensitive)
#   - severity_base: default severity before count adjustment
#   - description: human-readable category description
# ─────────────────────────────────────────────────────────────────

RISK_CATEGORIES = {
    "Competition": {
        "keywords": [
            "competition", "competitive", "competitor", "market share",
            "pricing pressure", "price war", "rival", "market saturation",
            "commoditization", "disruption", "disruptor",
        ],
        "severity_base": "Medium",
        "description": "Competitive threats from peers or new entrants",
    },
    "Regulatory": {
        "keywords": [
            "regulation", "regulatory", "compliance", "legal", "lawsuit",
            "litigation", "antitrust", "GDPR", "sanctions", "tariff",
            "tax reform", "government scrutiny", "SEC", "FTC", "penalty",
            "fine", "audit",
        ],
        "severity_base": "High",
        "description": "Legal, regulatory, or compliance risks",
    },
    "Rising Costs": {
        "keywords": [
            "cost increase", "inflation", "wage", "labor cost", "input cost",
            "operating expense", "COGS increase", "energy cost",
            "cost pressure", "margin compression", "higher costs",
            "cost headwind",
        ],
        "severity_base": "Medium",
        "description": "Input cost inflation and operating expense increases",
    },
    "Margin Pressure": {
        "keywords": [
            "margin pressure", "gross margin decline", "margin headwind",
            "compressed margin", "profitability concern", "EBITDA decline",
            "operating leverage", "deleveraging", "margin contraction",
        ],
        "severity_base": "High",
        "description": "Pressure on gross or operating margins",
    },
    "Supply Chain": {
        "keywords": [
            "supply chain", "shortage", "inventory", "logistics",
            "procurement", "supplier", "lead time", "backlog",
            "component shortage", "disruption", "delay",
        ],
        "severity_base": "Medium",
        "description": "Supply chain disruptions, shortages, or delays",
    },
    "Economic Slowdown": {
        "keywords": [
            "recession", "slowdown", "macro", "macroeconomic", "downturn",
            "contraction", "GDP", "consumer spending", "unemployment",
            "demand weakness", "softening demand", "economic uncertainty",
        ],
        "severity_base": "High",
        "description": "Macroeconomic slowdown or consumer demand weakness",
    },
    "Forex / Currency": {
        "keywords": [
            "foreign exchange", "FX", "currency headwind", "dollar strength",
            "exchange rate", "currency risk", "translation impact",
        ],
        "severity_base": "Low",
        "description": "Currency fluctuation and translation risks",
    },
    "Cybersecurity": {
        "keywords": [
            "cybersecurity", "data breach", "ransomware", "hack",
            "security incident", "cyber attack", "data privacy",
        ],
        "severity_base": "High",
        "description": "Cybersecurity threats and data privacy concerns",
    },
}


def detect_risks(sentences: List[str]) -> Dict:
    """
    Scan all transcript sentences for risk signals.

    Args:
        sentences: List of transcript sentences

    Returns:
        Dict containing:
          - risks: List of risk dicts (category, severity, count, examples)
          - risk_df_data: List of dicts ready for DataFrame display
          - total_risk_mentions: int
          - risk_score: float (0-100, used in confidence scoring)
    """
    # Initialize counters and example collectors
    risk_data = {
        category: {"count": 0, "sentences": []}
        for category in RISK_CATEGORIES
    }

    for sentence in sentences:
        sentence_lower = sentence.lower()

        for category, config in RISK_CATEGORIES.items():
            matches = sum(
                1 for kw in config["keywords"]
                if kw.lower() in sentence_lower
            )
            if matches > 0:
                risk_data[category]["count"] += 1
                # Store up to 3 example sentences per category
                if len(risk_data[category]["sentences"]) < 3:
                    risk_data[category]["sentences"].append(sentence.strip())

    # Build output structure
    risks = []
    for category, config in RISK_CATEGORIES.items():
        count = risk_data[category]["count"]
        if count == 0:
            continue

        # Adjust severity based on mention count
        base = config["severity_base"]
        if count >= 5:
            severity = "High"
        elif count >= 2:
            severity = "Medium" if base != "High" else "High"
        else:
            severity = base

        risks.append({
            "category": category,
            "description": config["description"],
            "count": count,
            "severity": severity,
            "example_sentences": risk_data[category]["sentences"],
            "example_sentence": risk_data[category]["sentences"][0] if risk_data[category]["sentences"] else "",
        })

    # Sort: High severity first, then by count
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    risks.sort(key=lambda x: (severity_order[x["severity"]], -x["count"]))

    total_mentions = sum(r["count"] for r in risks)

    # Risk score: scaled 0-100 for confidence formula
    # Higher risk mentions → higher risk score (subtracted in confidence)
    high_count = sum(r["count"] for r in risks if r["severity"] == "High")
    medium_count = sum(r["count"] for r in risks if r["severity"] == "Medium")
    raw_risk = (high_count * 2 + medium_count * 1)
    risk_score = min(100, raw_risk * 3)  # Scale and cap at 100

    return {
        "risks": risks,
        "total_risk_mentions": total_mentions,
        "risk_score": risk_score,
        "risk_categories_detected": len(risks),
    }
