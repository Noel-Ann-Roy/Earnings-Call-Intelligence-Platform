# Earnings Call Intelligence Platform
Built with Python | FinBERT | BERTopic | spaCy | Streamlit | Plotly
A production-quality NLP system that analyzes company earnings call transcripts
and generates analyst-style intelligence reports — entirely offline, using
open-source models. No paid APIs required.

---

## What This Project Does

Upload an earnings call transcript (TXT or PDF). The platform will automatically:

1. Extract and clean the text
2. Run financial sentiment analysis (FinBERT)
3. Model the major topics discussed (BERTopic)
4. Detect risk signals and growth signals (rule-based NLP)
5. Extract forward-looking guidance statements
6. Compute a Management Confidence Score (transparent formula)
7. Generate a full analyst report with a Bullish / Neutral / Bearish verdict

---

## Deep-Dive: How FinBERT Works

### What Is BERT?

BERT (Bidirectional Encoder Representations from Transformers) is a language
model pre-trained by Google on massive text corpora. It reads text
bidirectionally — meaning it considers words to the left AND right of each
word simultaneously, giving it deep contextual understanding.

### What Is FinBERT?

FinBERT (from ProsusAI) is BERT fine-tuned specifically on:
- Financial news articles
- Earnings call transcripts  
- Analyst reports

Because of this fine-tuning, FinBERT understands financial language far better
than a general-purpose sentiment model. For example:

  "Revenue growth decelerated to 12%"
  → General model: POSITIVE (growth)
  → FinBERT: NEGATIVE (deceleration)

### How FinBERT Classifies Sentiment

FinBERT takes a sentence and outputs three probability scores:
  - Positive (management is optimistic)
  - Neutral  (factual statement)
  - Negative (management is cautious or bearish)

The scores sum to 1.0. For example:
  Input:  "We expect strong revenue growth next quarter."
  Output: Positive=0.91, Neutral=0.07, Negative=0.02

In this project, we:
1. Split the transcript into individual sentences
2. Run FinBERT on each sentence
3. Average the scores across all sentences
4. Report the overall sentiment and confidence

### Why Not Train a Custom Model?

Training is unnecessary because:
- FinBERT was trained on ~50,000 financial sentences
- It already generalizes well to earnings call language
- Training would require labeled datasets, GPU time, and expertise
- Using a pre-trained model gives immediate, production-quality results

---

## Deep-Dive: How BERTopic Works

### The Problem BERTopic Solves

Given 500 sentences from an earnings call, which major themes are discussed?
You could read every sentence manually — or let BERTopic find the themes
automatically in seconds.

### BERTopic's Four-Step Pipeline

**Step 1: Sentence Embeddings**
Each sentence is converted to a dense numerical vector (embedding) using a
pre-trained Sentence Transformer model (all-MiniLM-L6-v2). Semantically
similar sentences produce similar vectors.

**Step 2: Dimensionality Reduction (UMAP)**
Embedding vectors are 384 dimensions. UMAP compresses them to 2-5 dimensions
while preserving neighborhood structure (similar sentences stay close together).

**Step 3: Clustering (HDBSCAN)**
HDBSCAN groups the low-dimensional points into clusters. Each cluster becomes
a topic. HDBSCAN is density-based — it doesn't require you to specify the
number of topics in advance.

**Step 4: Topic Representation (c-TF-IDF)**
For each cluster, BERTopic computes c-TF-IDF to find the words that best
describe THAT cluster vs. all other clusters. These become the topic keywords.

### Example Output

  Topic 0 (35%): revenue, growth, quarter, guidance, beat
  Topic 1 (22%): AI, cloud, platform, investment, adoption
  Topic 2 (18%): supply, chain, cost, inflation, margin
  Topic -1 (25%): Outlier/noise sentences (normal for BERTopic)

### Why Not Use LDA Instead?

BERTopic is more accurate than traditional Latent Dirichlet Allocation (LDA)
because it uses contextual embeddings, not just word frequencies.

---

## Project Structure

```
earnings_call_platform/
├── app.py                        # Main Streamlit application
├── requirements.txt              # All Python dependencies
├── setup.py                      # One-command setup helper
├── README.md                     # This file
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── extractor.py          # TXT + PDF text extraction
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── sentiment.py          # FinBERT sentiment analysis
│   │   └── topics.py             # BERTopic topic modeling
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── risks.py              # Rule-based risk detection
│   │   ├── growth.py             # Rule-based growth signal detection
│   │   ├── guidance.py           # Forward-looking statement extraction
│   │   ├── confidence.py         # Management Confidence Score
│   │   └── summary.py            # Executive summary generator
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── charts.py             # All Plotly chart functions
│   └── utils/
│       ├── __init__.py
│       └── text_utils.py         # Sentence splitting, cleaning
│
├── data/
│   └── samples/
│       └── sample_transcript.txt # Sample Apple earnings call
│
├── tests/
│   └── test_pipeline.py          # Unit tests
│
└── docs/
    └── ARCHITECTURE.md           # System design document
```

---

## Installation Instructions (Fresh Machine)

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip (comes with Python)
- ~4 GB disk space (for model downloads)
- ~4 GB RAM minimum (8 GB recommended)
- Internet connection for first run (model downloads)

### Step-by-Step Setup

**1. Clone or download the project folder**

```bash
cd earnings_call_platform
```

**2. Create a virtual environment (strongly recommended)**

```bash
python -m venv venv
```

Activate it:
- Windows:  venv\Scripts\activate
- Mac/Linux: source venv/bin/activate

**3. Install all dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the spaCy English model**

```bash
python -m spacy download en_core_web_sm
```

**5. First run (models auto-download)**

```bash
streamlit run app.py
```

On the very first run, two models download automatically:
- `ProsusAI/finbert` (~440 MB) from HuggingFace Hub
- `sentence-transformers/all-MiniLM-L6-v2` (~80 MB) for BERTopic

These are cached locally after the first download. All subsequent runs are
fully offline.

---

## How to Test with Real Transcripts

You can find free earnings call transcripts at:
- https://www.fool.com/earnings-call-transcripts/
- https://seekingalpha.com/earnings/earnings-transcripts
- https://ir.company.com (most companies' investor relations pages)
- SEC EDGAR: https://www.sec.gov/cgi-bin/browse-edgar

Search for: "Apple earnings call transcript Q4 2023"

Download as TXT or copy-paste into a .txt file, then upload to the platform.

---

## Datasets for Future Evaluation

- **ECTSum**: ~2,000 earnings call transcripts with human-written summaries
  (https://github.com/rajdeep345/ECTSum)
- **FinSBD**: Financial sentence boundary detection dataset
- **FPB (Financial PhraseBank)**: 4,840 labeled financial sentences
  (positive/neutral/negative) — perfect for evaluating FinBERT accuracy

---

## Management Confidence Score Formula

```
Score = (Sentiment × 0.40) + (Growth × 0.30) + (Guidance × 0.20) - (Risk × 0.10)

Where:
  Sentiment  = FinBERT positive ratio, scaled 0-100
  Growth     = Number of growth signals detected, capped and scaled 0-100
  Guidance   = Number of guidance statements detected, scaled 0-100
  Risk       = Number of high-severity risks detected, scaled 0-100 (subtracted)
```

---

## Final Verdict Logic

  Score ≥ 65 → BULLISH  (Strong management confidence)
  Score 40–64 → NEUTRAL (Mixed signals)
  Score < 40  → BEARISH  (Cautious or weak tone)
