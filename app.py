
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Earnings Call Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": """
        # 📈 Earnings Call Intelligence Platform
        
        **Built by:** Noel Ann Roy
        
        **Version:** 1.0.0
        
        **Tech Stack:**
        - FinBERT (ProsusAI) — Financial Sentiment Analysis
        - BERTopic — Topic Modeling  
        - spaCy — NLP Processing
        - Streamlit — Dashboard
        
        **GitHub:** https://github.com/yourusername/earnings-call-intelligence
        
        ---
        *Built as a portfolio project. Not investment advice.*
        """
    }
)

st.markdown("""
<style>
/* Global dark background */
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
[data-testid="stSidebar"] * {
    color: #E6EDF3 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 12px;
}
[data-testid="stMetricValue"] {
    color: #F0B90B !important;
    font-family: monospace !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8B949E !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Section headers */
h1, h2, h3 {
    color: #E6EDF3 !important;
    font-family: monospace !important;
    border-bottom: 1px solid #30363D;
    padding-bottom: 8px;
}

/* Info boxes */
.info-box {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-left: 4px solid #F0B90B;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 10px 0;
    font-family: monospace;
    font-size: 0.85rem;
    color: #E6EDF3;
}

/* BULLISH / NEUTRAL / BEARISH verdict badge */
.verdict-bullish {
    background-color: #0A2D0A;
    border: 2px solid #00C805;
    color: #00C805;
    font-family: monospace;
    font-size: 1.4rem;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 8px;
    display: inline-block;
    letter-spacing: 0.1em;
}
.verdict-neutral {
    background-color: #1A1400;
    border: 2px solid #FFA500;
    color: #FFA500;
    font-family: monospace;
    font-size: 1.4rem;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 8px;
    display: inline-block;
    letter-spacing: 0.1em;
}
.verdict-bearish {
    background-color: #2D0A0A;
    border: 2px solid #FF3B3B;
    color: #FF3B3B;
    font-family: monospace;
    font-size: 1.4rem;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 8px;
    display: inline-block;
    letter-spacing: 0.1em;
}

/* Risk severity badges */
.badge-high   { color: #FF3B3B; font-weight: bold; }
.badge-medium { color: #FFA500; font-weight: bold; }
.badge-low    { color: #58A6FF; font-weight: bold; }

/* Formula display */
.formula-box {
    background-color: #0D1117;
    border: 1px solid #F0B90B;
    border-radius: 6px;
    padding: 14px;
    font-family: monospace;
    color: #F0B90B;
    font-size: 0.9rem;
    margin: 10px 0;
}

/* Dataframe tables */
.stDataFrame {
    border: 1px solid #30363D !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 1px dashed #30363D;
    border-radius: 8px;
    padding: 10px;
}

/* Buttons */
.stButton > button {
    background-color: #161B22;
    color: #F0B90B;
    border: 1px solid #F0B90B;
    border-radius: 6px;
    font-family: monospace;
}
.stButton > button:hover {
    background-color: #F0B90B;
    color: #0D1117;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #30363D;
    border-radius: 6px;
    background-color: #161B22;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading FinBERT sentiment model...")
def load_sentiment_model():
    """Load and cache FinBERT. First call downloads ~440MB from HuggingFace."""
    from nlp.sentiment import FinBERTAnalyzer
    return FinBERTAnalyzer()


@st.cache_resource(show_spinner="Loading BERTopic model...")
def load_topic_modeler():
    """Load and cache BERTopic + sentence transformer."""
    from nlp.topics import TopicModeler
    return TopicModeler()

def render_sidebar():
    """Render sidebar with upload, settings, and about info."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <span style="font-size:2rem;">📈</span><br>
            <span style="font-family:monospace; font-size:1.1rem; color:#F0B90B; font-weight:bold;">
                EARNINGS CALL<br>INTELLIGENCE
            </span><br>
            <span style="font-size:0.7rem; color:#8B949E; letter-spacing:0.1em;">
                POWERED BY FINBERT + BERTOPIC
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📂 Upload Transcript")

        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload",
            type=["txt", "pdf"],
            help="Supported formats: TXT, PDF. Max size: 200MB",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")

        run_topics = st.checkbox(
            "Topic Modeling (BERTopic)",
            value=True,
            help="Disable to speed up analysis. BERTopic takes 10-30 seconds.",
        )
        min_topic_size = st.slider(
            "Min sentences per topic",
            min_value=3, max_value=15, value=5,
            help="Lower = more granular topics. Higher = fewer, broader topics.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="font-size:0.75rem; color:#8B949E; font-family:monospace;">
        <b>Models Used:</b><br>
        • ProsusAI/finbert (sentiment)<br>
        • all-MiniLM-L6-v2 (embeddings)<br>
        • spaCy en_core_web_sm (NLP)<br><br>
        <b>Analysis:</b><br>
        • Rule-based risk/growth extraction<br>
        • Transparent confidence scoring<br><br>
        <b>No paid APIs. Fully local.</b>
        </div>
        """, unsafe_allow_html=True)

    return uploaded_file, run_topics, min_topic_size


def run_analysis(text: str, run_topics: bool, min_topic_size: int) -> dict:
    """
    Orchestrate the full NLP pipeline.

    Order matters:
      1. Text preprocessing (sentence splitting)
      2. Sentiment analysis (FinBERT — most critical)
      3. Risk & growth extraction (fast, rule-based)
      4. Guidance extraction (fast, rule-based)
      5. Topic modeling (slow, ~10-30s — run last)
      6. Confidence score (aggregates all above)
      7. Executive summary (formatted narrative)
    """
    from utils.text_utils import split_into_sentences, get_word_count, get_paragraph_count
    from analysis.risks import detect_risks
    from analysis.growth import detect_growth_signals
    from analysis.guidance import extract_guidance
    from analysis.confidence import compute_confidence_score
    from analysis.summary import generate_executive_summary

    results = {}

    # ── Step 1: Text Stats ────────────────────────────────────────────────────
    results["word_count"] = get_word_count(text)
    results["paragraph_count"] = get_paragraph_count(text)

    # ── Step 2: Sentence Splitting ────────────────────────────────────────────
    with st.spinner(" Splitting transcript into sentences (spaCy)..."):
        sentences = split_into_sentences(text)
    results["sentences"] = sentences
    results["sentence_count"] = len(sentences)

    # ── Step 3: Sentiment Analysis ────────────────────────────────────────────
    with st.spinner(f" Running FinBERT on {len(sentences)} sentences..."):
        analyzer = load_sentiment_model()
        results["sentiment"] = analyzer.analyze(sentences)

    # ── Step 4: Risk Detection ────────────────────────────────────────────────
    with st.spinner(" Detecting risk signals..."):
        results["risks"] = detect_risks(sentences)

    # ── Step 5: Growth Detection ──────────────────────────────────────────────
    with st.spinner("📈 Detecting growth signals..."):
        results["growth"] = detect_growth_signals(sentences)

    # ── Step 6: Guidance Extraction ───────────────────────────────────────────
    with st.spinner(" Extracting forward-looking guidance..."):
        results["guidance"] = extract_guidance(sentences)

    # ── Step 7: Topic Modeling ────────────────────────────────────────────────
    if run_topics:
        with st.spinner(" Running BERTopic (this may take 15-30 seconds)..."):
            modeler = load_topic_modeler()
            modeler.min_topic_size = min_topic_size
            results["topics"] = modeler.fit_transform(sentences)
    else:
        from nlp.topics import TopicModeler
        results["topics"] = TopicModeler._empty_result()

    # ── Step 8: Confidence Score ──────────────────────────────────────────────
    with st.spinner(" Computing Management Confidence Score..."):
        results["confidence"] = compute_confidence_score(
            results["sentiment"],
            results["growth"],
            results["guidance"],
            results["risks"],
        )

    # ── Step 9: Executive Summary ─────────────────────────────────────────────
    with st.spinner(" Generating executive summary..."):
        results["summary"] = generate_executive_summary(
            results["sentiment"],
            results["growth"],
            results["risks"],
            results["guidance"],
            results["confidence"],
        )

    return results

def render_kpi_bar(results: dict):
    """Top-row KPI cards."""
    from visualization.charts import ACCENT_GOLD

    sentiment = results["sentiment"]
    confidence = results["confidence"]
    risks = results["risks"]
    growth = results["growth"]

    cols = st.columns(6)
    with cols[0]:
        st.metric("Confidence Score", f"{confidence['score']:.0f}/100")
    with cols[1]:
        st.metric("Sentiment", sentiment["overall_sentiment"],
                  f"{sentiment['confidence']:.0f}% conf.")
    with cols[2]:
        st.metric("Growth Signals", growth["categories_detected"],
                  f"{growth['strong_signals']} strong")
    with cols[3]:
        st.metric("Risk Categories", risks["risk_categories_detected"],
                  f"{risks['total_risk_mentions']} mentions")
    with cols[4]:
        st.metric("Guidance Stmts", results["guidance"]["total_statements"])
    with cols[5]:
        st.metric("Sentences Analyzed", results["sentence_count"])


def render_verdict_banner(confidence: dict):
    """Big verdict badge."""
    verdict = confidence["verdict"]
    score = confidence["score"]
    icon = confidence["verdict_icon"]
    css_class = f"verdict-{verdict.lower()}"

    st.markdown(f"""
    <div style="text-align:center; margin: 20px 0;">
        <span class="{css_class}">{icon} {verdict} — {score:.0f}/100</span>
    </div>
    """, unsafe_allow_html=True)


def render_sentiment_section(sentiment: dict):
    """Sentiment analysis section with gauge + pie + formula."""
    from visualization.charts import sentiment_gauge_chart, sentiment_pie_chart

    st.markdown("##  Sentiment Analysis (FinBERT)")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(sentiment_gauge_chart(sentiment), use_container_width=True)
    with col2:
        st.plotly_chart(sentiment_pie_chart(sentiment), use_container_width=True)

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.metric("Positive", f"{sentiment['positive_score']:.1%}")
    with col4:
        st.metric("Neutral", f"{sentiment['neutral_score']:.1%}")
    with col5:
        st.metric("Negative", f"{sentiment['negative_score']:.1%}")
    with col6:
        st.metric("Analyzed", f"{sentiment['total_sentences_analyzed']} sentences")

    # Sample sentence-level results
    with st.expander(" Sentence-Level Sentiment (top 20)", expanded=False):
        if sentiment["sentence_results"]:
            df = pd.DataFrame(sentiment["sentence_results"][:20])[
                ["sentence", "label", "positive", "neutral", "negative"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_risk_section(risks: dict):
    """Risk detection section with table and chart."""
    from visualization.charts import risk_severity_chart

    st.markdown("##  Risk Detection")

    risk_list = risks["risks"]
    if not risk_list:
        st.info("No significant risk signals detected in this transcript.")
        return

    col1, col2 = st.columns([1.2, 1])
    with col1:
        # Risk table
        rows = []
        for r in risk_list:
            sev = r["severity"]
            badge_class = f"badge-{sev.lower()}"
            rows.append({
                "Category": r["category"],
                "Severity": r["severity"],
                "Mentions": r["count"],
                "Description": r["description"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col2:
        st.plotly_chart(risk_severity_chart(risk_list), use_container_width=True)

    # Evidence sentences
    with st.expander(" Evidence Sentences", expanded=False):
        for r in risk_list[:5]:  # Top 5 risks
            if r.get("example_sentences"):
                st.markdown(f"**{r['category']}** ({r['severity']})")
                for sent in r["example_sentences"][:2]:
                    st.markdown(f"""
                    <div class="info-box">"{sent}"</div>
                    """, unsafe_allow_html=True)


def render_growth_section(growth: dict):
    """Growth signal section."""
    from visualization.charts import growth_signal_chart

    st.markdown("##  Growth Signal Detection")

    signals = growth["signals"]
    if not signals:
        st.info("No significant growth signals detected.")
        return

    col1, col2 = st.columns([1.2, 1])
    with col1:
        rows = [{
            "Category": s["category"],
            "Strength": f"{s['strength_icon']} {s['strength']}",
            "Mentions": s["count"],
            "Description": s["description"],
        } for s in signals]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col2:
        st.plotly_chart(growth_signal_chart(signals), use_container_width=True)

    with st.expander(" Evidence Sentences", expanded=False):
        for s in signals[:5]:
            if s.get("example_sentences"):
                st.markdown(f"**{s['category']}** ({s['strength']})")
                for sent in s["example_sentences"][:2]:
                    st.markdown(f"""
                    <div class="info-box">"{sent}"</div>
                    """, unsafe_allow_html=True)


def render_guidance_section(guidance: dict):
    """Forward-looking guidance section."""
    from visualization.charts import guidance_tone_pie

    st.markdown("##  Guidance & Forward-Looking Statements")

    stmts = guidance["statements"]
    if not stmts:
        st.info("No forward-looking guidance statements detected.")
        return

    col1, col2 = st.columns([1.5, 1])
    with col1:
        rows = [{
            "Type": s["type"],
            "Tone": s["tone"],
            "Statement": s["sentence"][:200],
        } for s in stmts[:15]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col2:
        st.plotly_chart(
            guidance_tone_pie(guidance["tone_counts"]),
            use_container_width=True
        )


def render_topic_section(topics: dict):
    """Topic modeling section."""
    from visualization.charts import topic_bar_chart

    st.markdown("##  Topic Modeling (BERTopic)")

    if not topics["topics"]:
        st.info(
            "No topics extracted. The transcript may be too short, or topic modeling "
            "was disabled. Try reducing 'Min sentences per topic' in the sidebar."
        )
        return

    col1, col2 = st.columns([1.2, 1])
    with col1:
        rows = [{
            "Topic": t["label"],
            "Keywords": t["keywords_str"],
            "Sentences": t["count"],
            "Share": f"{t['percentage']:.1f}%",
        } for t in topics["topics"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col2:
        st.plotly_chart(topic_bar_chart(topics["topics"]), use_container_width=True)

    st.caption(
        f"ℹ BERTopic identified {topics['total_topics']} topics. "
        f"{topics['outlier_percentage']:.0f}% of sentences were classified as outliers (Topic -1 — normal)."
    )


def render_confidence_section(confidence: dict):
    """Confidence score section with formula and breakdown."""
    from visualization.charts import confidence_gauge_chart, confidence_breakdown_chart

    st.markdown("##  Management Confidence Score")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            confidence_gauge_chart(confidence["score"], confidence["verdict"]),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            confidence_breakdown_chart(confidence["components"]),
            use_container_width=True,
        )

    # Formula
    st.markdown("""
    <div class="formula-box">
    <b>FORMULA:</b><br>
    Score = (Sentiment × 0.40) + (Growth × 0.30) + (Guidance × 0.20) - (Risk × 0.10)<br><br>
    """ + confidence["formula_display"] + """
    </div>
    """, unsafe_allow_html=True)


def render_summary_section(summary: str):
    """Executive summary section."""
    st.markdown("##  Executive Summary")
    st.markdown("""
    <div class="info-box">
     This is an automated NLP analysis. It is NOT investment advice.
    Always verify findings with primary sources and human judgment.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(summary)


def generate_report_text(results: dict) -> str:
    """Generate a plain-text analyst report for download."""
    lines = [
        "=" * 70,
        "EARNINGS CALL INTELLIGENCE PLATFORM — ANALYST REPORT",
        "=" * 70,
        "",
        f"VERDICT: {results['confidence']['verdict']}",
        f"MANAGEMENT CONFIDENCE SCORE: {results['confidence']['score']}/100",
        "",
        "FORMULA:",
        results["confidence"]["formula_display"].replace("**", ""),
        "",
        "─" * 70,
        "EXECUTIVE SUMMARY",
        "─" * 70,
        "",
        results["summary"].replace("##", "").replace("**", "").replace("*", ""),
        "",
        "─" * 70,
        "SENTIMENT ANALYSIS (FinBERT)",
        "─" * 70,
        f"Overall Sentiment:  {results['sentiment']['overall_sentiment']}",
        f"Confidence:         {results['sentiment']['confidence']:.1f}%",
        f"Positive Score:     {results['sentiment']['positive_score']:.4f}",
        f"Neutral Score:      {results['sentiment']['neutral_score']:.4f}",
        f"Negative Score:     {results['sentiment']['negative_score']:.4f}",
        f"Sentences Analyzed: {results['sentiment']['total_sentences_analyzed']}",
        "",
        "─" * 70,
        "RISK SIGNALS",
        "─" * 70,
    ]

    for r in results["risks"]["risks"]:
        lines.append(f"[{r['severity']:6}] {r['category']:20} | {r['count']} mentions")
        if r.get("example_sentence"):
            lines.append(f"         → {r['example_sentence'][:150]}")

    lines += [
        "",
        "─" * 70,
        "GROWTH SIGNALS",
        "─" * 70,
    ]
    for s in results["growth"]["signals"]:
        lines.append(f"[{s['strength']:8}] {s['category']:20} | {s['count']} mentions")

    lines += [
        "",
        "─" * 70,
        "FORWARD GUIDANCE STATEMENTS",
        "─" * 70,
    ]
    for stmt in results["guidance"]["statements"][:20]:
        lines.append(f"[{stmt['type']:22}] [{stmt['tone']:10}] {stmt['sentence'][:180]}")

    lines += [
        "",
        "─" * 70,
        "TOPICS (BERTopic)",
        "─" * 70,
    ]
    for t in results["topics"]["topics"]:
        lines.append(f"{t['label']:10} {t['percentage']:5.1f}%  Keywords: {t['keywords_str']}")

    lines += [
        "",
        "=" * 70,
        "DISCLAIMER: Automated NLP analysis. Not investment advice.",
        "=" * 70,
    ]

    return "\n".join(lines)



def render_landing():
    """Shown when no file is uploaded yet."""
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <h1 style="color:#F0B90B; font-family:monospace; border:none; font-size:2.5rem;">
            📈 EARNINGS CALL INTELLIGENCE
        </h1>
        <p style="color:#8B949E; font-family:monospace; font-size:1rem;">
            Upload an earnings call transcript to generate an analyst-style intelligence report.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box">
        <b> FinBERT Sentiment</b><br><br>
        Financial sentiment analysis using ProsusAI/finbert — fine-tuned BERT 
        that understands the difference between "revenue decelerated" (negative) 
        and "revenue grew" (positive).
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
        <b> BERTopic Modeling</b><br><br>
        Discover the major themes discussed without specifying topics in advance.
        Uses UMAP + HDBSCAN + c-TF-IDF to cluster semantically similar sentences.
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-box">
        <b> Transparent Scoring</b><br><br>
        Every number is explained. The Management Confidence Score uses a 
        documented formula: Sentiment (40%) + Growth (30%) + Guidance (20%) 
        − Risk (10%).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top: 30px; color:#8B949E; font-family:monospace; font-size:0.8rem;">
    Supported formats: .txt, .pdf &nbsp;|&nbsp; 
    Models: FinBERT, BERTopic, spaCy &nbsp;|&nbsp; 
    No paid APIs &nbsp;|&nbsp; Fully local
    </div>
    """, unsafe_allow_html=True)


# ─── Main App Entry Point ─────────────────────────────────────────────────────

def main():
    uploaded_file, run_topics, min_topic_size = render_sidebar()

    if uploaded_file is None:
        render_landing()
        return

    from ingestion.extractor import extract_text

    try:
        with st.spinner(f" Extracting text from {uploaded_file.name}..."):
            text = extract_text(uploaded_file)
    except Exception as e:
        st.error(f" Failed to extract text: {e}")
        return

    if len(text.strip()) < 100:
        st.error("The uploaded file appears to be empty or too short. Please upload a real earnings call transcript.")
        return

    st.success(f" Extracted {len(text):,} characters from {uploaded_file.name}")

    # ── Run Pipeline ─────────────────────────────────────────────────────────
    with st.spinner("Running analysis pipeline..."):
        try:
            results = run_analysis(text, run_topics, min_topic_size)
        except Exception as e:
            st.error(f" Analysis failed: {e}")
            st.exception(e)
            return

    st.markdown("---")
    render_verdict_banner(results["confidence"])
    render_kpi_bar(results)
    st.markdown("---")

    render_summary_section(results["summary"])
    st.markdown("---")

    render_sentiment_section(results["sentiment"])
    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        render_risk_section(results["risks"])
    with col_right:
        render_growth_section(results["growth"])

    st.markdown("---")
    render_guidance_section(results["guidance"])
    st.markdown("---")

    if run_topics:
        render_topic_section(results["topics"])
        st.markdown("---")

    render_confidence_section(results["confidence"])
    st.markdown("---")

    st.markdown("## 📥 Download Report")
    report_text = generate_report_text(results)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label=" Download Analyst Report (.txt)",
            data=report_text,
            file_name="earnings_intelligence_report.txt",
            mime="text/plain",
        )
    with col_dl2:
        st.download_button(
            label=" Download Summary (.md)",
            data=results["summary"],
            file_name="executive_summary.md",
            mime="text/markdown",
        )

    # ── Raw Transcript Preview ─────────────────────────────────────────────────
    with st.expander("📄 Raw Transcript Preview (first 3000 chars)", expanded=False):
        st.text(text[:3000])


if __name__ == "__main__":
    main()
