"""
src/visualization/charts.py
============================
Phase 7: All Plotly chart functions for the dashboard.

Uses Plotly for interactive, Bloomberg-terminal-inspired visuals.
All functions return Plotly figure objects that Streamlit displays
with st.plotly_chart().

COLOR PALETTE (Bloomberg-inspired dark theme):
  Background:   #0D1117 (near-black)
  Panel:        #161B22
  Accent Gold:  #F0B90B
  Green:        #00C805
  Red:          #FF3B3B
  Text:         #E6EDF3
  Muted:        #8B949E
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd

# ─── Theme Constants ──────────────────────────────────────────────────────────
BG_COLOR = "#0D1117"
PANEL_COLOR = "#161B22"
ACCENT_GOLD = "#F0B90B"
GREEN = "#00C805"
RED = "#FF3B3B"
BLUE = "#58A6FF"
ORANGE = "#FFA500"
TEXT_COLOR = "#E6EDF3"
MUTED_COLOR = "#8B949E"

BASE_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=PANEL_COLOR,
    font=dict(color=TEXT_COLOR, family="monospace"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ─── 1. Sentiment Gauge Chart ─────────────────────────────────────────────────

def sentiment_gauge_chart(sentiment_result: Dict) -> go.Figure:
    """
    Gauge chart showing overall sentiment score from -100 to +100.

    We map FinBERT scores to a signed scale:
      score = (positive - negative) × 100
    """
    pos = sentiment_result.get("positive_score", 0.33)
    neg = sentiment_result.get("negative_score", 0.33)
    score = (pos - neg) * 100  # Range: -100 to +100

    # Color based on score
    if score > 20:
        bar_color = GREEN
    elif score < -20:
        bar_color = RED
    else:
        bar_color = ORANGE

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 36, "color": TEXT_COLOR}},
        title={"text": "Sentiment Score", "font": {"color": MUTED_COLOR, "size": 14}},
        gauge={
            "axis": {
                "range": [-100, 100],
                "tickcolor": MUTED_COLOR,
                "tickfont": {"color": MUTED_COLOR, "size": 10},
            },
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": PANEL_COLOR,
            "bordercolor": MUTED_COLOR,
            "steps": [
                {"range": [-100, -20], "color": "#2D0A0A"},   # Dark red
                {"range": [-20, 20],   "color": "#1A1A00"},   # Dark yellow
                {"range": [20, 100],   "color": "#0A2D0A"},   # Dark green
            ],
            "threshold": {
                "line": {"color": ACCENT_GOLD, "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))

    fig.update_layout(**BASE_LAYOUT, height=280)
    return fig


# ─── 2. Confidence Score Gauge ────────────────────────────────────────────────

def confidence_gauge_chart(score: float, verdict: str) -> go.Figure:
    """Gauge chart for the Management Confidence Score (0-100)."""
    color_map = {"BULLISH": GREEN, "NEUTRAL": ORANGE, "BEARISH": RED}
    bar_color = color_map.get(verdict, ORANGE)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 50, "increasing": {"color": GREEN}, "decreasing": {"color": RED}},
        number={"font": {"size": 40, "color": TEXT_COLOR}},
        title={"text": f"Confidence Score — {verdict}", "font": {"color": bar_color, "size": 14}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 20, 40, 65, 80, 100],
                "ticktext": ["0", "20", "BEARISH", "BULLISH", "80", "100"],
                "tickfont": {"color": MUTED_COLOR, "size": 9},
            },
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": PANEL_COLOR,
            "bordercolor": MUTED_COLOR,
            "steps": [
                {"range": [0, 40],   "color": "#2D0A0A"},
                {"range": [40, 65],  "color": "#1A1A00"},
                {"range": [65, 100], "color": "#0A2D0A"},
            ],
        },
    ))

    fig.update_layout(**BASE_LAYOUT, height=300)
    return fig


# ─── 3. Sentiment Breakdown Pie Chart ────────────────────────────────────────

def sentiment_pie_chart(sentiment_result: Dict) -> go.Figure:
    """Pie chart showing positive / neutral / negative distribution."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        sentiment_result.get("positive_score", 0.33),
        sentiment_result.get("neutral_score", 0.34),
        sentiment_result.get("negative_score", 0.33),
    ]
    colors = [GREEN, BLUE, RED]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
        textfont=dict(color=TEXT_COLOR, size=12),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Score: %{value:.3f}<br>Share: %{percent}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Sentiment Distribution", font=dict(color=MUTED_COLOR, size=13)),
        showlegend=False,
        height=280,
        annotations=[dict(
            text=sentiment_result.get("overall_sentiment", "Neutral"),
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=TEXT_COLOR),
        )],
    )
    return fig


# ─── 4. Topic Bar Chart ───────────────────────────────────────────────────────

def topic_bar_chart(topics: List[Dict]) -> go.Figure:
    """Horizontal bar chart showing topic percentages."""
    if not topics:
        return _empty_figure("No topics detected")

    labels = [f"Topic {i+1}: {', '.join(t['keywords'][:3])}" for i, t in enumerate(topics)]
    values = [t["percentage"] for t in topics]
    colors = [ACCENT_GOLD, BLUE, GREEN, ORANGE, "#C084FC", "#FB923C", "#34D399"][:len(topics)]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color=BG_COLOR, width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=11),
        hovertemplate="<b>%{y}</b><br>Share: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Top Discussed Topics", font=dict(color=MUTED_COLOR, size=13)),
        xaxis=dict(
            title="% of Sentences",
            tickfont=dict(color=MUTED_COLOR),
            gridcolor="#21262D",
            showgrid=True,
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color=TEXT_COLOR, size=10)),
        height=max(300, len(topics) * 55 + 100),
    )
    return fig


# ─── 5. Risk Severity Chart ───────────────────────────────────────────────────

def risk_severity_chart(risks: List[Dict]) -> go.Figure:
    """Horizontal bar chart of risks colored by severity."""
    if not risks:
        return _empty_figure("No risks detected")

    severity_colors = {"High": RED, "Medium": ORANGE, "Low": BLUE}

    categories = [r["category"] for r in risks]
    counts = [r["count"] for r in risks]
    colors = [severity_colors.get(r["severity"], MUTED_COLOR) for r in risks]

    fig = go.Figure(go.Bar(
        x=counts,
        y=categories,
        orientation="h",
        marker=dict(color=colors, line=dict(color=BG_COLOR, width=1)),
        text=[f'{r["severity"]} ({r["count"]})' for r in risks],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=10),
        hovertemplate="<b>%{y}</b><br>Mentions: %{x}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Risk Signals by Category", font=dict(color=MUTED_COLOR, size=13)),
        xaxis=dict(
            title="Mention Count",
            tickfont=dict(color=MUTED_COLOR),
            gridcolor="#21262D",
            showgrid=True,
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color=TEXT_COLOR, size=10)),
        height=max(280, len(risks) * 50 + 100),
    )
    return fig


# ─── 6. Growth Signal Chart ───────────────────────────────────────────────────

def growth_signal_chart(signals: List[Dict]) -> go.Figure:
    """Bar chart of growth signals colored by strength."""
    if not signals:
        return _empty_figure("No growth signals detected")

    strength_colors = {"Strong": GREEN, "Moderate": BLUE, "Emerging": MUTED_COLOR}

    categories = [s["category"] for s in signals]
    counts = [s["count"] for s in signals]
    colors = [strength_colors.get(s["strength"], BLUE) for s in signals]

    fig = go.Figure(go.Bar(
        x=counts,
        y=categories,
        orientation="h",
        marker=dict(color=colors, line=dict(color=BG_COLOR, width=1)),
        text=[f'{s["strength"]} ({s["count"]})' for s in signals],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=10),
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Growth Signals by Category", font=dict(color=MUTED_COLOR, size=13)),
        xaxis=dict(
            title="Mention Count",
            tickfont=dict(color=MUTED_COLOR),
            gridcolor="#21262D",
            showgrid=True,
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(color=TEXT_COLOR, size=10)),
        height=max(280, len(signals) * 50 + 100),
    )
    return fig


# ─── 7. Confidence Score Component Breakdown ──────────────────────────────────

def confidence_breakdown_chart(components: Dict) -> go.Figure:
    """Stacked bar showing contribution of each component to confidence score."""
    comp_labels = []
    contributions = []
    bar_colors = []

    color_map = {
        "sentiment": BLUE,
        "growth": GREEN,
        "guidance": ACCENT_GOLD,
        "risk": RED,
    }

    for key, data in components.items():
        comp_labels.append(data["label"])
        contributions.append(abs(data["contribution"]))
        bar_colors.append(color_map.get(key, MUTED_COLOR))

    fig = go.Figure(go.Bar(
        x=comp_labels,
        y=contributions,
        marker=dict(color=bar_colors, line=dict(color=BG_COLOR, width=1)),
        text=[f"{c:+.1f}" for c in [components[k]["contribution"] for k in components]],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=11),
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Score Component Breakdown", font=dict(color=MUTED_COLOR, size=13)),
        xaxis=dict(tickfont=dict(color=TEXT_COLOR)),
        yaxis=dict(
            title="Points Contributed",
            tickfont=dict(color=MUTED_COLOR),
            gridcolor="#21262D",
        ),
        height=300,
    )
    return fig


# ─── 8. Guidance Tone Pie ─────────────────────────────────────────────────────

def guidance_tone_pie(tone_counts: Dict) -> go.Figure:
    """Pie chart of guidance tone distribution."""
    labels = list(tone_counts.keys())
    values = list(tone_counts.values())
    colors = [GREEN, RED, BLUE]

    if sum(values) == 0:
        return _empty_figure("No guidance found")

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
        textfont=dict(color=TEXT_COLOR, size=12),
        textinfo="label+percent",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Guidance Tone Distribution", font=dict(color=MUTED_COLOR, size=13)),
        showlegend=False,
        height=260,
    )
    return fig


# ─── Helper ───────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(color=MUTED_COLOR, size=14),
    )
    fig.update_layout(**BASE_LAYOUT, height=200)
    return fig
