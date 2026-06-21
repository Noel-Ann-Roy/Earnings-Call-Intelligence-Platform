"""
src/analysis/growth.py
======================
Phase 5b: Rule-Based Growth Signal Detection

APPROACH:
  Same keyword-matching approach as risk detection, but tuned for positive
  growth indicators. Growth signals include product launches, revenue beats,
  customer expansion, AI adoption, and strategic partnerships.

WHY RULE-BASED?
  Growth signals are consistently expressed using a specific vocabulary
  in earnings calls. Terms like "record revenue", "strong adoption",
  "new partnership", "expanded into" are reliable growth indicators.

SIGNAL STRENGTH:
  Strong: 3+ mentions — management emphasizes this as a key driver
  Moderate: 2 mentions
  Emerging: 1 mention — early signal, worth watching
"""

from typing import Dict, List

GROWTH_CATEGORIES = {
    "Revenue Growth": {
        "keywords": [
            "revenue growth", "revenue increased", "top-line growth",
            "record revenue", "revenue beat", "strong revenue",
            "revenue acceleration", "revenue momentum", "sales growth",
            "exceeded expectations", "above guidance",
        ],
        "description": "Revenue and top-line growth signals",
    },
    "Customer Growth": {
        "keywords": [
            "customer growth", "new customers", "customer acquisition",
            "customer retention", "net new customers", "user growth",
            "active users", "subscriber growth", "MAU", "DAU",
            "customer base expanded", "record customers",
        ],
        "description": "Customer and user base expansion",
    },
    "Product Expansion": {
        "keywords": [
            "new product", "product launch", "product expansion",
            "innovation", "product roadmap", "new feature",
            "platform expansion", "product portfolio", "new offering",
            "launched", "released", "introduced",
        ],
        "description": "New product launches and feature releases",
    },
    "Market Expansion": {
        "keywords": [
            "new market", "market expansion", "geographic expansion",
            "international growth", "emerging market", "new geography",
            "expanded into", "market penetration", "addressable market",
            "TAM", "new vertical",
        ],
        "description": "Geographic and vertical market expansion",
    },
    "AI & Technology": {
        "keywords": [
            "artificial intelligence", "AI", "machine learning", "ML",
            "generative AI", "large language model", "LLM", "automation",
            "digital transformation", "cloud adoption", "platform AI",
            "AI-powered", "AI-driven", "AI integration",
        ],
        "description": "AI adoption, technology innovation, and digital transformation",
    },
    "Partnerships": {
        "keywords": [
            "partnership", "strategic alliance", "collaboration",
            "joint venture", "acquisition", "M&A", "integration",
            "ecosystem", "channel partner", "OEM agreement",
        ],
        "description": "Strategic partnerships and M&A activity",
    },
    "Margin Improvement": {
        "keywords": [
            "margin expansion", "margin improvement", "gross margin increase",
            "operating leverage", "efficiency gains", "cost optimization",
            "profitability improvement", "EBITDA growth", "margin tailwind",
        ],
        "description": "Margin expansion and profitability improvements",
    },
    "Cash Flow": {
        "keywords": [
            "free cash flow", "FCF", "cash generation", "buyback",
            "dividend", "share repurchase", "cash position", "balance sheet strength",
            "debt reduction", "capital return",
        ],
        "description": "Strong cash flow and shareholder returns",
    },
}

SIGNAL_STRENGTH_MAP = {
    "Strong": "🟢",
    "Moderate": "🟡",
    "Emerging": "🔵",
}


def detect_growth_signals(sentences: List[str]) -> Dict:
    """
    Scan transcript sentences for growth signals.

    Args:
        sentences: List of transcript sentences

    Returns:
        Dict containing:
          - signals: List of growth signal dicts
          - total_signals: int
          - growth_score: float (0-100, used in confidence formula)
    """
    signal_data = {
        category: {"count": 0, "sentences": []}
        for category in GROWTH_CATEGORIES
    }

    for sentence in sentences:
        sentence_lower = sentence.lower()

        for category, config in GROWTH_CATEGORIES.items():
            matches = sum(
                1 for kw in config["keywords"]
                if kw.lower() in sentence_lower
            )
            if matches > 0:
                signal_data[category]["count"] += 1
                if len(signal_data[category]["sentences"]) < 3:
                    signal_data[category]["sentences"].append(sentence.strip())

    signals = []
    for category, config in GROWTH_CATEGORIES.items():
        count = signal_data[category]["count"]
        if count == 0:
            continue

        # Assign signal strength
        if count >= 3:
            strength = "Strong"
        elif count >= 2:
            strength = "Moderate"
        else:
            strength = "Emerging"

        signals.append({
            "category": category,
            "description": config["description"],
            "count": count,
            "strength": strength,
            "strength_icon": SIGNAL_STRENGTH_MAP[strength],
            "example_sentences": signal_data[category]["sentences"],
            "example_sentence": signal_data[category]["sentences"][0] if signal_data[category]["sentences"] else "",
        })

    # Sort: Strong first, then by count
    strength_order = {"Strong": 0, "Moderate": 1, "Emerging": 2}
    signals.sort(key=lambda x: (strength_order[x["strength"]], -x["count"]))

    total_signals = sum(s["count"] for s in signals)
    strong_count = sum(1 for s in signals if s["strength"] == "Strong")

    # Growth score for confidence formula
    # Each strong signal contributes more weight
    raw_score = (strong_count * 15) + (len(signals) * 5) + (total_signals * 2)
    growth_score = min(100, raw_score)

    return {
        "signals": signals,
        "total_signals": total_signals,
        "growth_score": growth_score,
        "strong_signals": strong_count,
        "categories_detected": len(signals),
    }
