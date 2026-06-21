import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


# ─── Test: Text Extraction ────────────────────────────────────────────────────

class TestExtractor:
    def test_txt_extraction(self):
        """Test that TXT files are extracted correctly."""
        from src.ingestion.extractor import extract_text_from_txt

        sample = b"Apple Inc reported strong revenue growth of 15 percent. The CEO expressed confidence in future outlook."
        result = extract_text_from_txt(sample)
        assert len(result) > 50
        assert "Apple" in result
        assert "revenue" in result

    def test_empty_txt(self):
        """Empty file should return empty string."""
        from src.ingestion.extractor import extract_text_from_txt
        result = extract_text_from_txt(b"")
        assert result == ""

    def test_clean_text_removes_page_markers(self):
        """Page markers should be stripped."""
        from src.ingestion.extractor import extract_text_from_txt
        text = b"Revenue grew 10 percent. Page 3 of 15. Supply chain improved."
        result = extract_text_from_txt(text)
        assert "Page 3 of 15" not in result


# ─── Test: Sentence Splitting ─────────────────────────────────────────────────

class TestTextUtils:
    def test_sentence_splitting(self):
        """Should split paragraph into multiple sentences."""
        from src.utils.text_utils import split_into_sentences

        text = (
            "We expect revenue to grow 15 percent next quarter. "
            "The supply chain has normalized significantly. "
            "Our AI investments are paying off with strong adoption."
        )
        sentences = split_into_sentences(text)
        assert len(sentences) >= 2  # Should split into at least 2

    def test_word_count(self):
        from src.utils.text_utils import get_word_count
        assert get_word_count("hello world foo bar") == 4

    def test_truncate_text(self):
        from src.utils.text_utils import truncate_text
        long_text = "word " * 200
        result = truncate_text(long_text, max_chars=50)
        assert len(result) <= 50


# ─── Test: Risk Detection ─────────────────────────────────────────────────────

class TestRiskDetection:
    def _get_sentences(self):
        return [
            "We face significant competition from local players in China.",
            "Regulatory scrutiny from the FTC and antitrust investigations remain a concern.",
            "Rising labor costs are putting pressure on our operating margins.",
            "Supply chain disruptions have caused component shortages and delays.",
            "The macroeconomic slowdown may reduce consumer spending.",
            "Revenue grew 15 percent year over year.",  # Not a risk sentence
        ]

    def test_detects_competition_risk(self):
        from src.analysis.risks import detect_risks
        sentences = self._get_sentences()
        result = detect_risks(sentences)
        categories = [r["category"] for r in result["risks"]]
        assert "Competition" in categories

    def test_detects_regulatory_risk(self):
        from src.analysis.risks import detect_risks
        sentences = self._get_sentences()
        result = detect_risks(sentences)
        categories = [r["category"] for r in result["risks"]]
        assert "Regulatory" in categories

    def test_risk_score_is_numeric(self):
        from src.analysis.risks import detect_risks
        result = detect_risks(self._get_sentences())
        assert 0 <= result["risk_score"] <= 100

    def test_empty_sentences(self):
        from src.analysis.risks import detect_risks
        result = detect_risks([])
        assert result["risks"] == []
        assert result["total_risk_mentions"] == 0


# ─── Test: Growth Detection ───────────────────────────────────────────────────

class TestGrowthDetection:
    def _get_sentences(self):
        return [
            "Revenue growth accelerated to 20 percent year over year.",
            "We added 5 million new customers in the quarter.",
            "Our AI-powered platform is seeing strong adoption.",
            "We launched three new products this quarter including our flagship device.",
            "We expanded into 12 new international markets.",
            "Free cash flow generation remains strong.",
        ]

    def test_detects_revenue_growth(self):
        from src.analysis.growth import detect_growth_signals
        result = detect_growth_signals(self._get_sentences())
        categories = [s["category"] for s in result["signals"]]
        assert "Revenue Growth" in categories

    def test_detects_ai_signal(self):
        from src.analysis.growth import detect_growth_signals
        result = detect_growth_signals(self._get_sentences())
        categories = [s["category"] for s in result["signals"]]
        assert "AI & Technology" in categories

    def test_growth_score_range(self):
        from src.analysis.growth import detect_growth_signals
        result = detect_growth_signals(self._get_sentences())
        assert 0 <= result["growth_score"] <= 100


# ─── Test: Guidance Extraction ────────────────────────────────────────────────

class TestGuidanceExtraction:
    def test_detects_forward_looking_statements(self):
        from src.analysis.guidance import extract_guidance

        sentences = [
            "We expect revenue to be between 89 billion and 92 billion next quarter.",
            "We anticipate gross margin to expand by 50 basis points.",
            "Going forward, we forecast strong free cash flow generation.",
            "The weather was nice today.",  # Not guidance
        ]
        result = extract_guidance(sentences)
        assert result["total_statements"] >= 2

    def test_classifies_revenue_guidance(self):
        from src.analysis.guidance import extract_guidance

        sentences = ["We expect revenue to grow 15 percent in the next fiscal year."]
        result = extract_guidance(sentences)
        types = [s["type"] for s in result["statements"]]
        assert "Revenue Guidance" in types

    def test_no_guidance_in_factual_text(self):
        from src.analysis.guidance import extract_guidance

        sentences = [
            "Revenue was 89.5 billion in Q4.",
            "Gross margin was 45.2 percent.",
        ]
        result = extract_guidance(sentences)
        # Factual sentences without trigger phrases should not be flagged
        assert result["total_statements"] == 0


# ─── Test: Confidence Score ───────────────────────────────────────────────────

class TestConfidenceScore:
    def _mock_inputs(self):
        sentiment = {
            "positive_score": 0.70,
            "negative_score": 0.10,
            "neutral_score": 0.20,
            "overall_sentiment": "Positive",
            "confidence": 70.0,
        }
        growth = {"growth_score": 65.0, "categories_detected": 4, "strong_signals": 2}
        guidance = {"guidance_score": 55.0, "total_statements": 6, "tone_counts": {"Optimistic": 4, "Cautious": 1, "Neutral": 1}}
        risks = {"risk_score": 30.0, "risk_categories_detected": 2, "total_risk_mentions": 5, "risks": []}
        return sentiment, growth, guidance, risks

    def test_score_in_range(self):
        from src.analysis.confidence import compute_confidence_score
        s, g, gu, r = self._mock_inputs()
        result = compute_confidence_score(s, g, gu, r)
        assert 0 <= result["score"] <= 100

    def test_high_sentiment_gives_high_score(self):
        from src.analysis.confidence import compute_confidence_score
        s, g, gu, r = self._mock_inputs()
        s["positive_score"] = 0.95
        s["negative_score"] = 0.02
        result = compute_confidence_score(s, g, gu, r)
        assert result["score"] > 50  # Should be bullish

    def test_verdict_is_valid(self):
        from src.analysis.confidence import compute_confidence_score
        s, g, gu, r = self._mock_inputs()
        result = compute_confidence_score(s, g, gu, r)
        assert result["verdict"] in ["BULLISH", "NEUTRAL", "BEARISH"]

    def test_formula_display_not_empty(self):
        from src.analysis.confidence import compute_confidence_score
        s, g, gu, r = self._mock_inputs()
        result = compute_confidence_score(s, g, gu, r)
        assert len(result["formula_display"]) > 10


# ─── Integration Test: Full Pipeline with Sample Transcript ───────────────────

class TestFullPipeline:
    """End-to-end test using the sample transcript."""

    @pytest.fixture
    def transcript_text(self):
        sample_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "samples", "sample_transcript.txt"
        )
        with open(sample_path, "r") as f:
            return f.read()

    def test_sentence_split_sample(self, transcript_text):
        from src.utils.text_utils import split_into_sentences
        sentences = split_into_sentences(transcript_text)
        assert len(sentences) > 20, f"Expected >20 sentences, got {len(sentences)}"

    def test_risk_detection_sample(self, transcript_text):
        from src.utils.text_utils import split_into_sentences
        from src.analysis.risks import detect_risks

        sentences = split_into_sentences(transcript_text)
        result = detect_risks(sentences)
        # Sample transcript explicitly mentions competition, regulatory, costs
        assert result["risk_categories_detected"] >= 2

    def test_growth_detection_sample(self, transcript_text):
        from src.utils.text_utils import split_into_sentences
        from src.analysis.growth import detect_growth_signals

        sentences = split_into_sentences(transcript_text)
        result = detect_growth_signals(sentences)
        assert result["categories_detected"] >= 2

    def test_guidance_extraction_sample(self, transcript_text):
        from src.utils.text_utils import split_into_sentences
        from src.analysis.guidance import extract_guidance

        sentences = split_into_sentences(transcript_text)
        result = extract_guidance(sentences)
        # The sample has multiple "we expect", "we anticipate" etc.
        assert result["total_statements"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
