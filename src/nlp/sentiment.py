"""
src/nlp/sentiment.py
====================
Phase 3: Financial Sentiment Analysis using FinBERT

MODEL: ProsusAI/finbert
  - Fine-tuned BERT for financial sentiment
  - 3-class output: positive / neutral / negative
  - Trained on ~50,000 labeled financial sentences

HOW IT WORKS:
  1. Load model and tokenizer from HuggingFace (cached after first download)
  2. For each sentence in the transcript:
     a. Tokenize (convert words to integer token IDs)
     b. Run forward pass through BERT transformer layers
     c. Apply softmax to the 3 output logits → probabilities
  3. Average probabilities across all sentences
  4. Return aggregate scores + overall label

TOKENIZATION LIMIT:
  BERT models have a hard limit of 512 tokens per input.
  We truncate each sentence to 512 tokens (roughly 400 words).
  For earnings calls, sentences rarely exceed this.

WHY NOT TRAIN CUSTOM?
  - FinBERT already achieves ~85% accuracy on financial PhraseBank dataset
  - Custom training needs labeled data, GPU, and weeks of work
  - Transfer learning from FinBERT is the industry standard approach
"""

from typing import Dict, List
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# FinBERT model identifier on HuggingFace Hub
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Labels in the order FinBERT outputs them
# (This order matches the model's label2id mapping)
LABELS = ["positive", "negative", "neutral"]


class FinBERTAnalyzer:
    """
    Wraps ProsusAI/finbert for sentence-level financial sentiment analysis.

    Usage:
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze(sentences)
    """

    def __init__(self):
        """
        Load the FinBERT tokenizer and model.

        On first call: downloads ~440MB from HuggingFace and caches locally.
        On subsequent calls: loads from cache (fast, offline).
        """
        self.tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        self.model.eval()  # Set to evaluation mode (disables dropout)

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _predict_single(self, sentence: str) -> Dict[str, float]:
        """
        Run FinBERT on a single sentence.

        Args:
            sentence: A single sentence string (max ~400 words)

        Returns:
            Dict with keys "positive", "negative", "neutral" and float values
        """
        # Tokenize: convert text to token IDs
        # truncation=True ensures we never exceed 512 tokens
        # return_tensors="pt" gives PyTorch tensors
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass — no gradient needed for inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.logits shape: [1, 3]  (batch_size=1, num_classes=3)
        # Softmax converts raw logits to probabilities that sum to 1.0
        probs = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

        # FinBERT's label order: positive=0, negative=1, neutral=2
        return {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2]),
        }

    def analyze(self, sentences: List[str], batch_size: int = 16) -> Dict:
        """
        Analyze a list of sentences and return aggregate sentiment scores.

        We process in batches for efficiency.

        Args:
            sentences: List of sentence strings from the transcript
            batch_size: Number of sentences per batch (reduce if OOM)

        Returns:
            Dict containing:
              - positive_score: float (0-1)
              - negative_score: float (0-1)
              - neutral_score:  float (0-1)
              - overall_sentiment: str ("Positive" / "Neutral" / "Negative")
              - confidence: float (0-100, percentage of top label)
              - sentence_results: List of per-sentence dicts (for detail view)
        """
        if not sentences:
            return self._empty_result()

        all_scores = []
        sentence_results = []

        # Process sentences in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            for sentence in batch:
                scores = self._predict_single(sentence)
                all_scores.append(scores)
                # Determine label for this sentence
                label = max(scores, key=scores.get)
                sentence_results.append({
                    "sentence": sentence[:200],  # Truncate for display
                    "positive": round(scores["positive"], 3),
                    "negative": round(scores["negative"], 3),
                    "neutral": round(scores["neutral"], 3),
                    "label": label.capitalize(),
                })

        # Aggregate: simple mean across all sentences
        avg_positive = float(np.mean([s["positive"] for s in all_scores]))
        avg_negative = float(np.mean([s["negative"] for s in all_scores]))
        avg_neutral = float(np.mean([s["neutral"] for s in all_scores]))

        # Determine overall sentiment (whichever label has the highest average)
        scores_dict = {
            "Positive": avg_positive,
            "Negative": avg_negative,
            "Neutral": avg_neutral,
        }
        overall = max(scores_dict, key=scores_dict.get)
        confidence = scores_dict[overall] * 100  # Convert to percentage

        return {
            "positive_score": round(avg_positive, 4),
            "negative_score": round(avg_negative, 4),
            "neutral_score": round(avg_neutral, 4),
            "overall_sentiment": overall,
            "confidence": round(confidence, 1),
            "sentence_results": sentence_results,
            "total_sentences_analyzed": len(sentences),
        }

    @staticmethod
    def _empty_result() -> Dict:
        """Return a safe empty result if no sentences provided."""
        return {
            "positive_score": 0.33,
            "negative_score": 0.33,
            "neutral_score": 0.34,
            "overall_sentiment": "Neutral",
            "confidence": 34.0,
            "sentence_results": [],
            "total_sentences_analyzed": 0,
        }
