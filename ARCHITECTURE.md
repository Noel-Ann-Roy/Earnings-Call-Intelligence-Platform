# System Architecture — Earnings Call Intelligence Platform

## Data Flow

```
User uploads .txt / .pdf
         │
         ▼
┌─────────────────────┐
│  ingestion/         │
│  extractor.py       │  pdfplumber / utf-8 decode → clean text string
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  utils/             │
│  text_utils.py      │  spaCy sentence splitter → List[str]
└─────────────────────┘
         │
    ┌────┴────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
┌──────────────┐                   ┌──────────────────┐
│ nlp/         │                   │ analysis/        │
│ sentiment.py │                   │ risks.py         │
│ (FinBERT)    │                   │ growth.py        │
│              │                   │ guidance.py      │
│ → +/n/- pct  │                   │ (rule-based)     │
└──────────────┘                   └──────────────────┘
    │                                         │
    │             ┌──────────────┐            │
    │             │ nlp/         │            │
    │             │ topics.py    │            │
    │             │ (BERTopic)   │            │
    │             └──────────────┘            │
    │                    │                   │
    └──────────┬──────────┘                   │
               │                              │
               ▼                              ▼
         ┌─────────────────────────────────────┐
         │     analysis/confidence.py          │
         │     Score = (S×0.4)+(G×0.3)+        │
         │            (Gu×0.2)-(R×0.1)         │
         └─────────────────────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────────────┐
         │     analysis/summary.py             │
         │     Template → Executive Summary    │
         └─────────────────────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────────────┐
         │     visualization/charts.py         │
         │     Plotly figures                  │
         └─────────────────────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────────────┐
         │     app.py (Streamlit)              │
         │     Dashboard rendered in browser   │
         └─────────────────────────────────────┘
```

## Model Details

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| ProsusAI/finbert | ~440MB | Financial sentiment (3-class) | HuggingFace |
| all-MiniLM-L6-v2 | ~80MB | Sentence embeddings for BERTopic | HuggingFace |
| en_core_web_sm | ~12MB | Sentence boundary detection | spaCy |

All models are cached after first download.
No internet connection needed after initial setup.

## Performance Expectations

| Operation | Time (CPU) |
|-----------|-----------|
| Text extraction | < 1s |
| Sentence splitting (500 sentences) | 2-5s |
| FinBERT (500 sentences) | 30-90s |
| Risk/Growth extraction | < 1s |
| BERTopic | 15-45s |
| Confidence score | < 1s |
| **Total** | **1-3 minutes** |

With GPU (CUDA), FinBERT runs ~10x faster.

## Why Rule-Based for Risk/Growth/Guidance?

ML classifiers require labeled training data. Creating high-quality
labeled data for financial NLP is expensive and time-consuming.

Rule-based systems offer:
- Zero training data required
- Full interpretability (you can trace every result to a keyword)
- Easy to extend (add new keywords without retraining)
- Industry-standard for structured information extraction

Financial terms are highly standardized — "we expect", "we anticipate",
"margin pressure", "competitive headwind" have consistent meanings across
earnings calls from different companies.

## Extending the Platform

**Add new risk categories**: Edit `RISK_CATEGORIES` dict in `risks.py`
**Add new growth signals**: Edit `GROWTH_CATEGORIES` dict in `growth.py`
**Change confidence weights**: Edit `WEIGHTS` dict in `confidence.py`
**Improve BERTopic**: Try `all-mpnet-base-v2` embeddings (larger, slower)
**Add speaker attribution**: Parse speaker labels before sentence splitting
**Historical comparison**: Store results in SQLite, compare across quarters
