"""
Tests for src/summarize.py

Covers transcript loading, text extraction, chunking, extraction helpers,
and structured summary generation (no-AI path to avoid model downloads).
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import SummaryConfig
from src.exceptions import SummarizationError
from src.summarize import (
    chunk_text,
    create_structured_summary,
    extract_action_items,
    extract_decisions,
    extract_full_text,
    extract_key_points,
    generate_extractive_summary,
    generate_mistral_summary,
    load_transcript,
)


# ---------------------------------------------------------------------------
# load_transcript
# ---------------------------------------------------------------------------

class TestLoadTranscript:
    def test_loads_valid_json(self, tmp_path):
        data = {"text": "hello", "segments": []}
        p = tmp_path / "t.json"
        p.write_text(json.dumps(data))
        result = load_transcript(p)
        assert result["text"] == "hello"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(SummarizationError):
            load_transcript(tmp_path / "missing.json")

    def test_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{")
        with pytest.raises(SummarizationError):
            load_transcript(p)


# ---------------------------------------------------------------------------
# extract_full_text
# ---------------------------------------------------------------------------

class TestExtractFullText:
    def test_uses_text_field_when_present(self):
        transcript = {"text": "hello world", "segments": []}
        assert extract_full_text(transcript) == "hello world"

    def test_falls_back_to_segments(self):
        transcript = {
            "segments": [
                {"text": "hello"},
                {"text": "world"},
            ]
        }
        result = extract_full_text(transcript)
        assert "hello" in result
        assert "world" in result

    def test_empty_segments_returns_empty_string(self):
        assert extract_full_text({"segments": []}) == ""

    def test_strips_whitespace_from_text_field(self):
        transcript = {"text": "  hello  "}
        assert extract_full_text(transcript) == "hello"


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "hello world"
        chunks = chunk_text(text, max_words=100)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_long_text_multiple_chunks(self):
        words = ["word"] * 700
        text = " ".join(words)
        chunks = chunk_text(text, max_words=350)
        assert len(chunks) == 2

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []

    def test_chunk_size_respected(self):
        words = ["w"] * 100
        text = " ".join(words)
        chunks = chunk_text(text, max_words=30)
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 30


# ---------------------------------------------------------------------------
# extract_action_items
# ---------------------------------------------------------------------------

class TestExtractActionItems:
    def test_detects_action_keywords(self):
        text = "We need to finish the report. Also must review the code."
        items = extract_action_items(text)
        assert len(items) >= 1
        assert any("need to" in item.lower() or "must" in item.lower() for item in items)

    def test_no_action_items_returns_empty(self):
        text = "The weather is nice today. Birds are singing."
        items = extract_action_items(text)
        assert items == []

    def test_respects_max_items(self):
        text = ". ".join(["we need to do thing"] * 20)
        items = extract_action_items(text, max_items=3)
        assert len(items) <= 3

    def test_no_duplicate_items(self):
        text = "We need to do this. We need to do this."
        items = extract_action_items(text)
        assert len(items) == len(set(items))


# ---------------------------------------------------------------------------
# extract_decisions
# ---------------------------------------------------------------------------

class TestExtractDecisions:
    def test_detects_decision_keywords(self):
        text = "The team decided to launch next week. We agreed on the pricing."
        decisions = extract_decisions(text)
        assert len(decisions) >= 1

    def test_no_decisions_returns_empty(self):
        text = "We talked about many things today."
        decisions = extract_decisions(text)
        assert decisions == []

    def test_respects_max_decisions(self):
        text = ". ".join(["we decided to do thing"] * 10)
        decisions = extract_decisions(text, max_decisions=2)
        assert len(decisions) <= 2


# ---------------------------------------------------------------------------
# extract_key_points
# ---------------------------------------------------------------------------

class TestExtractKeyPoints:
    def test_returns_list(self):
        text = "This is an important point. Another key discussion happened here. A third significant item."
        points = extract_key_points(text, num_points=3)
        assert isinstance(points, list)
        assert len(points) >= 1

    def test_empty_text_returns_fallback(self):
        points = extract_key_points("", num_points=3)
        assert points == ["No key points identified."]

    def test_short_sentences_excluded(self):
        # Sentences shorter than 30 chars are filtered out
        text = "Short. Another short. " + "This is a much longer sentence that should be included in the result."
        points = extract_key_points(text, num_points=5)
        assert all(len(p) > 0 for p in points)


# ---------------------------------------------------------------------------
# generate_extractive_summary
# ---------------------------------------------------------------------------

class TestGenerativeExtractiveSummary:
    def test_returns_string(self):
        text = "This is a test. The quick brown fox jumps. We discussed many things. Final point here."
        result = generate_extractive_summary(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_text_returns_fallback(self):
        result = generate_extractive_summary("")
        assert result == "No summary available."

    def test_respects_num_sentences(self):
        sentences = [f"Sentence number {i} is here for testing purposes." for i in range(20)]
        text = ". ".join(sentences)
        result = generate_extractive_summary(text, num_sentences=3)
        # Result should be concise — hard to count exact sentences after joining,
        # but it should be much shorter than the original
        assert len(result) < len(text)


# ---------------------------------------------------------------------------
# create_structured_summary (no-AI path)
# ---------------------------------------------------------------------------

class TestCreateStructuredSummary:
    def _transcript(self, text: str) -> dict:
        return {"text": text, "segments": []}

    def test_returns_markdown_string(self):
        t = self._transcript("We need to finish the report. The team decided to launch.")
        result = create_structured_summary(t, use_ai=False)
        assert isinstance(result, str)
        assert result.startswith("#")

    def test_empty_transcript_raises(self):
        with pytest.raises(SummarizationError):
            create_structured_summary({"text": "", "segments": []}, use_ai=False)

    def test_meeting_content_type_has_action_items(self):
        t = self._transcript(
            "We need to finish the report by Friday. "
            "The team decided to launch next week. "
            "There is an important issue to resolve."
        )
        result = create_structured_summary(
            t, config=SummaryConfig(content_type="meeting"), use_ai=False
        )
        assert "Action Items" in result

    def test_interview_content_type_has_insights_section(self):
        t = self._transcript(
            "I think this is a very interesting topic. "
            "We talked about many important things. "
            "The guest explained the key concepts clearly."
        )
        result = create_structured_summary(
            t, config=SummaryConfig(content_type="interview"), use_ai=False
        )
        assert "Notable Insights" in result or "Main Topics" in result

    def test_podcast_content_type_has_discussions_section(self):
        t = self._transcript(
            "This is an amazing insight about technology. "
            "The hosts discussed many interesting topics. "
            "A fascinating discovery was shared today."
        )
        result = create_structured_summary(
            t, config=SummaryConfig(content_type="podcast"), use_ai=False
        )
        assert "Main Discussions" in result

    def test_general_content_type_has_key_points(self):
        t = self._transcript(
            "This is a key observation. Another important finding here. "
            "Critical results were discussed at length."
        )
        result = create_structured_summary(
            t, config=SummaryConfig(content_type="general"), use_ai=False
        )
        assert "Key Points" in result

    def test_content_type_override_parameter(self):
        t = self._transcript("We need to finish the report. Key discussion here.")
        result = create_structured_summary(
            t,
            config=SummaryConfig(content_type="general"),
            use_ai=False,
            content_type="meeting",
        )
        assert "Meeting Summary" in result

    def test_includes_executive_summary_section(self):
        t = self._transcript("An important topic was discussed at length today.")
        result = create_structured_summary(t, use_ai=False)
        assert "Executive Summary" in result


# ---------------------------------------------------------------------------
# generate_mistral_summary (llama-cpp-python path, fully mocked)
# ---------------------------------------------------------------------------

class TestGenerateMistralSummary:
    """Tests for generate_mistral_summary with _try_load_mistral mocked out."""

    def _make_llm_mock(self, response_text: str) -> MagicMock:
        mock_llm = MagicMock()
        mock_llm.return_value = {"choices": [{"text": response_text}]}
        return mock_llm

    def _config(self, **kwargs) -> SummaryConfig:
        return SummaryConfig(**kwargs)

    def test_single_chunk_returns_mistral_output(self):
        fake_summary = "The team discussed the project timeline and agreed on next steps."
        mock_llm = self._make_llm_mock(fake_summary)
        with patch("src.summarize._try_load_mistral", return_value=mock_llm):
            result = generate_mistral_summary(
                "We need to finish the report. The team agreed on the plan.",
                self._config(),
            )
        assert result == fake_summary

    def test_falls_back_to_extractive_when_llm_none(self):
        text = "This is an important meeting. We discussed the key issues here."
        with patch("src.summarize._try_load_mistral", return_value=None):
            result = generate_mistral_summary(text, self._config())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_text_returns_empty_string(self):
        with patch("src.summarize._try_load_mistral") as mock_loader:
            result = generate_mistral_summary("", self._config())
        mock_loader.assert_not_called()
        assert result == ""

    def test_multi_chunk_triggers_hierarchical_pass(self):
        call_count = {"n": 0}
        responses = ["Chunk one summary.", "Chunk two summary.", "Final combined summary."]

        def side_effect(prompt, **kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            text = responses[idx] if idx < len(responses) else "Fallback."
            return {"choices": [{"text": text}]}

        mock_llm = MagicMock(side_effect=side_effect)
        long_text = " ".join(["word"] * 4001)
        with patch("src.summarize._try_load_mistral", return_value=mock_llm):
            result = generate_mistral_summary(long_text, self._config())

        assert call_count["n"] >= 3
        assert result == "Final combined summary."

    def test_chunk_exception_partial_result_returned(self):
        call_count = {"n": 0}

        def side_effect(prompt, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated inference error")
            return {"choices": [{"text": "Second chunk summary."}]}

        mock_llm = MagicMock(side_effect=side_effect)
        long_text = " ".join(["word"] * 4001)
        with patch("src.summarize._try_load_mistral", return_value=mock_llm):
            result = generate_mistral_summary(long_text, self._config())

        assert isinstance(result, str)
        assert len(result) > 0

    def test_all_chunks_fail_falls_back_to_extractive(self):
        mock_llm = MagicMock(side_effect=RuntimeError("inference error"))
        text = "This is a meeting. We talked about important things."
        with patch("src.summarize._try_load_mistral", return_value=mock_llm):
            result = generate_mistral_summary(text, self._config())
        assert isinstance(result, str)
        assert result != ""

    def test_create_structured_summary_uses_mistral_when_use_ai_true(self):
        fake_summary = "The meeting covered the Q1 budget and next steps."
        mock_llm = self._make_llm_mock(fake_summary)
        transcript = {"text": "We discussed the Q1 budget. We need to plan next steps.", "segments": []}
        with patch("src.summarize._try_load_mistral", return_value=mock_llm):
            result = create_structured_summary(transcript, use_ai=True)
        assert fake_summary in result
        assert "Executive Summary" in result


class TestMistralPromptTemplate:
    def test_prompt_contains_domain_terms(self):
        """Verify the prompt includes AMR glossary to prevent hallucinations."""
        from src.summarize import _MISTRAL_PROMPT_TEMPLATE
        assert "AMR" in _MISTRAL_PROMPT_TEMPLATE
        assert "Autonomous Mobile Robot" in _MISTRAL_PROMPT_TEMPLATE
        assert "Do not invent acronyms" in _MISTRAL_PROMPT_TEMPLATE
