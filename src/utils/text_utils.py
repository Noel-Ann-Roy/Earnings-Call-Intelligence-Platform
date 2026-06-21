"""
src/utils/text_utils.py
=======================
Shared text processing utilities used across all NLP modules.

Uses spaCy for linguistically accurate sentence splitting.
spaCy's sentence splitter is more robust than naive split-on-period
because it handles:
  - Abbreviations: "Q3 rev. grew 12% vs. $1.2B est."
  - Decimal numbers: "EPS of $3.14"
  - URLs and ticker symbols

Why spaCy?
  spaCy is an industrial-strength NLP library. Its en_core_web_sm model
  is only ~12MB but provides tokenization, POS tagging, and sentence
  boundary detection that handles financial text accurately.
"""

import re
from typing import List
import spacy

# Load spaCy model once at module import time.
# This avoids reloading the model on every function call.
# The model must be downloaded first: python -m spacy download en_core_web_sm
try:
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])
    # Increase max length for long transcripts (default is 1M chars)
    _nlp.max_length = 5_000_000
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


def split_into_sentences(text: str) -> List[str]:
    """
    Split transcript text into individual sentences using spaCy.

    For very long transcripts, we process in chunks to avoid memory issues.

    Args:
        text: Full transcript text

    Returns:
        List of sentence strings, cleaned of empty/very short ones
    """
    # Process in chunks of 50,000 characters for memory efficiency
    chunk_size = 50_000
    sentences = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i: i + chunk_size]
        doc = _nlp(chunk)
        for sent in doc.sents:
            s = sent.text.strip()
            # Filter out very short sentences (noise) and blank lines
            if len(s) > 10 and not s.isdigit():
                sentences.append(s)

    return sentences


def get_word_count(text: str) -> int:
    """Return the number of words in a text string."""
    return len(text.split())


def get_paragraph_count(text: str) -> int:
    """Return the number of non-empty paragraphs."""
    return len([p for p in text.split("\n\n") if p.strip()])


def truncate_text(text: str, max_chars: int = 512) -> str:
    """
    Truncate text to a maximum character count, ending at a word boundary.

    FinBERT has a 512-token input limit. This ensures we don't exceed it.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


def clean_sentence(sentence: str) -> str:
    """
    Light cleaning for a single sentence before NLP processing.
    Removes multiple spaces and normalizes punctuation.
    """
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = sentence.strip()
    return sentence
