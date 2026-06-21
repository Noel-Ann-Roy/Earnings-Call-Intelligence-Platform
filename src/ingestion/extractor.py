"""
src/ingestion/extractor.py
==========================
Phase 2: Data Ingestion

Handles extraction of raw text from:
  - .txt files (plain text)
  - .pdf files (using pdfplumber)

Why pdfplumber?
  It preserves text layout better than PyPDF2 for formatted PDFs.
  Earnings call PDFs often have two-column layouts, headers, and footers.
  pdfplumber handles these gracefully.
"""

import io
import re
import pdfplumber


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Decode raw bytes from a .txt upload.

    Args:
        file_bytes: Raw bytes from Streamlit's file_uploader

    Returns:
        Cleaned string of transcript text
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback for files with Latin-1 encoding
        text = file_bytes.decode("latin-1")

    return _clean_text(text)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF using pdfplumber.

    Iterates over every page, extracts text, and joins with newlines.

    Args:
        file_bytes: Raw bytes from Streamlit's file_uploader

    Returns:
        Cleaned string of transcript text
    """
    text_parts = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    raw_text = "\n".join(text_parts)
    return _clean_text(raw_text)


def extract_text(uploaded_file) -> str:
    """
    Unified entry point for text extraction.

    Detects file type by extension and routes to the correct extractor.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Cleaned transcript text as a string

    Raises:
        ValueError: If file type is unsupported
    """
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    elif filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}. Upload .txt or .pdf only.")


def _clean_text(text: str) -> str:
    """
    Normalize whitespace and remove common PDF artifacts.

    Steps:
      1. Replace Windows-style line endings
      2. Collapse multiple blank lines to one
      3. Strip leading/trailing whitespace per line
      4. Remove page number artifacts like "Page 3 of 15"
      5. Remove common operator interjection markers: "[Operator]", "(inaudible)"
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page markers
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    # Remove inaudible tags and similar artifacts
    text = re.sub(r"\(inaudible\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[inaudible\]", "", text, flags=re.IGNORECASE)

    # Remove lines that are just speaker labels (e.g. "OPERATOR:", "Analyst:")
    # We keep them as context but strip the label prefix for NLP
    # (Optional: uncomment to strip speaker labels)
    # text = re.sub(r"^[A-Z][A-Z\s]+:", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
