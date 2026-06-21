"""
src/nlp/topics.py
=================
Phase 4: Topic Modeling using BERTopic

MODEL: BERTopic with all-MiniLM-L6-v2 sentence embeddings

PIPELINE:
  1. SentenceTransformer ("all-MiniLM-L6-v2")
     → Converts sentences to 384-dim semantic vectors
     → ~80MB download, cached after first use

  2. UMAP (Uniform Manifold Approximation and Projection)
     → Reduces 384-dim vectors to 5-dim
     → Preserves local neighborhood structure
     → This is why semantically similar sentences cluster together

  3. HDBSCAN (Hierarchical Density-Based Spatial Clustering)
     → Groups the 5-dim points into clusters (= topics)
     → Topic -1 is the "outlier" bucket (normal and expected)
     → Does NOT require specifying number of topics in advance

  4. c-TF-IDF (Class-based Term Frequency - Inverse Document Frequency)
     → For each cluster, finds words that are uniquely frequent IN that cluster
       vs. all other clusters combined
     → These become the topic keywords

HYPERPARAMETERS CHOSEN FOR EARNINGS CALLS:
  - min_topic_size=5: A topic must have at least 5 sentences
    (prevents overfitting to single-sentence "topics")
  - nr_topics="auto": BERTopic merges similar topics automatically
  - top_n_words=8: Show 8 representative words per topic

WHY NOT LDA?
  LDA treats documents as bags of words — it ignores word order and context.
  BERTopic uses BERT embeddings so "revenue beat" and "earnings surprise" map
  to similar vectors even though they share no words.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


class TopicModeler:
    """
    BERTopic-based topic extraction for earnings call transcripts.

    Usage:
        modeler = TopicModeler()
        result = modeler.fit_transform(sentences)
    """

    def __init__(self, min_topic_size: int = 5, top_n_words: int = 8):
        """
        Initialize the topic modeler.

        Args:
            min_topic_size: Minimum sentences per topic (default 5)
            top_n_words: Number of keywords to show per topic (default 8)
        """
        self.min_topic_size = min_topic_size
        self.top_n_words = top_n_words

        # Load sentence transformer for embeddings
        # all-MiniLM-L6-v2 is the best balance of speed and accuracy
        # for short-to-medium length financial sentences
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # BERTopic model (not fitted yet — fitting happens in fit_transform)
        self._model = None

    def fit_transform(self, sentences: List[str]) -> Dict:
        """
        Fit BERTopic on the transcript sentences and extract topics.

        Args:
            sentences: List of cleaned sentence strings

        Returns:
            Dict containing:
              - topics: List of dicts with topic info
              - topic_df: pandas DataFrame of topics
              - assignments: List of topic IDs per sentence
        """
        if len(sentences) < 10:
            return self._empty_result()

        # Initialize BERTopic with our embedding model
        self._model = BERTopic(
            embedding_model=self._embedding_model,
            min_topic_size=self.min_topic_size,
            top_n_words=self.top_n_words,
            nr_topics="auto",       # Auto-merge similar topics
            verbose=False,          # Suppress training output
            calculate_probabilities=True,
        )

        # Fit and transform: this is the main BERTopic pipeline
        # topics: list of topic IDs, one per sentence (-1 = outlier)
        # probs:  confidence of each assignment
        topics, probs = self._model.fit_transform(sentences)

        return self._format_results(topics, probs, sentences)

    def _format_results(
        self,
        topics: List[int],
        probs: np.ndarray,
        sentences: List[str],
    ) -> Dict:
        """
        Format BERTopic output into clean, display-ready structures.
        """
        # Get topic info: counts and top words per topic
        topic_info = self._model.get_topic_info()

        # Filter out the outlier topic (-1)
        real_topics = topic_info[topic_info["Topic"] != -1]

        total_sentences = len(sentences)
        formatted_topics = []

        for _, row in real_topics.iterrows():
            topic_id = row["Topic"]
            count = row["Count"]
            percentage = round((count / total_sentences) * 100, 1)

            # Get top words for this topic
            words_scores = self._model.get_topic(topic_id)
            keywords = [word for word, _ in words_scores[:self.top_n_words]]

            formatted_topics.append({
                "topic_id": topic_id,
                "label": f"Topic {topic_id + 1}",
                "keywords": keywords,
                "keywords_str": ", ".join(keywords),
                "count": count,
                "percentage": percentage,
            })

        # Sort by percentage descending
        formatted_topics.sort(key=lambda x: x["percentage"], reverse=True)

        # Build DataFrame for display
        topic_df = pd.DataFrame(formatted_topics)[
            ["label", "keywords_str", "count", "percentage"]
        ] if formatted_topics else pd.DataFrame()

        # Outlier count
        outlier_count = len([t for t in topics if t == -1])
        outlier_pct = round((outlier_count / total_sentences) * 100, 1)

        return {
            "topics": formatted_topics,
            "topic_df": topic_df,
            "assignments": topics,
            "total_topics": len(formatted_topics),
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
        }

    @staticmethod
    def _empty_result() -> Dict:
        """Return empty result if not enough sentences."""
        return {
            "topics": [],
            "topic_df": pd.DataFrame(),
            "assignments": [],
            "total_topics": 0,
            "outlier_count": 0,
            "outlier_percentage": 0,
        }
