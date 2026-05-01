"""
Tests for all improvements introduced in this sprint:

  FIX 1  - Strict grounding prompt in LlamaService._build_messages()
  FIX 2  - Pre-flight similarity gate in the /query endpoint
  FIX 3  - ChromaService._normalize_query()
  FIX 4  - ScoringService._calibrate_score() and _query_complexity_weight()
  FIX 5  - ChromaService.search_with_rerank() fallback when reranker=None
"""
import math
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3 — _normalize_query
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeQuery:
    @pytest.fixture
    def chroma(self):
        from services.chroma_service import ChromaService
        svc = ChromaService.__new__(ChromaService)
        return svc

    def test_normalize_query_collapses_spaces(self, chroma):
        """'word 2 vec' → 'word2vec' (word+digit then digit+short-word collapse)."""
        result = chroma._normalize_query("word 2 vec")
        assert result == "word2vec"

    def test_normalize_query_lowercase(self, chroma):
        """'GPT 4 Model' → 'gpt4 model' (lowercased; 'model' ≥ 5 chars, not collapsed)."""
        result = chroma._normalize_query("GPT 4 Model")
        assert result == "gpt4 model"

    def test_normalize_strips_whitespace(self, chroma):
        result = chroma._normalize_query("  hello world  ")
        assert result == "hello world"

    def test_normalize_preserves_plain_text(self, chroma):
        """A query with no digits should not be altered (beyond lowercase)."""
        result = chroma._normalize_query("What is machine learning?")
        assert result == "what is machine learning?"

    def test_normalize_gpt4_style(self, chroma):
        """'gpt 4' → 'gpt4' (2-token word-digit collapse)."""
        result = chroma._normalize_query("gpt 4")
        assert result == "gpt4"


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — Pre-flight similarity gate in the /query endpoint
# ═══════════════════════════════════════════════════════════════════════════

class TestPreflightGate:
    def test_preflight_blocks_empty_passages(self, client, test_user):
        """Empty retrieval → LLM never called, confidence = 0.0."""
        with patch(
            "api.routes.endpoints.chroma_service.search_with_rerank",
            return_value=[],
        ), patch(
            "api.routes.endpoints.llama_service.generate_answer"
        ) as mock_llm:
            resp = client.post(
                "/api/query",
                json={"question": "What is the boiling point of unobtainium?"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence_score"] == 0.0
        assert "history_id" in body
        mock_llm.assert_not_called()

    def test_preflight_blocks_low_similarity(self, client, test_user):
        """Passages with max similarity 0.30 → LLM not called."""
        low_sim = [
            {
                "text":             "Unrelated document text.",
                "source":           "doc.pdf",
                "similarity_score": 0.30,
                "metadata":         {},
                "page":             0,
            }
        ]
        with patch(
            "api.routes.endpoints.chroma_service.search_with_rerank",
            return_value=low_sim,
        ), patch(
            "api.routes.endpoints.llama_service.generate_answer"
        ) as mock_llm:
            resp = client.post(
                "/api/query",
                json={"question": "Completely off-topic question?"},
            )

        assert resp.status_code == 200
        assert resp.json()["confidence_score"] == 0.0
        mock_llm.assert_not_called()

    def test_preflight_passes_sufficient_similarity(self, client, test_user):
        """Passages with similarity ≥ 0.45 → LLM IS called."""
        good_passage = [
            {
                "text":             "Python is a high-level programming language.",
                "source":           "prog.pdf",
                "similarity_score": 0.82,
                "metadata":         {},
                "page":             1,
            }
        ]
        with patch(
            "api.routes.endpoints.chroma_service.search_with_rerank",
            return_value=good_passage,
        ), patch(
            "api.routes.endpoints.llama_service.generate_answer",
            return_value="Python is a programming language.",
        ) as mock_llm, patch(
            "api.routes.endpoints.scoring_service.compute_confidence_score",
            return_value=(0.85, "Good match.", [], {}),
        ):
            resp = client.post(
                "/api/query",
                json={"question": "What is Python?"},
            )

        assert resp.status_code == 200
        mock_llm.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4 — _query_complexity_weight
# ═══════════════════════════════════════════════════════════════════════════

class TestQueryComplexityWeight:
    @pytest.fixture
    def scorer(self):
        from services.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        return svc

    def test_complexity_weight_one_word(self, scorer):
        """1-word query → weight = 0.70."""
        assert scorer._query_complexity_weight("Python") == 0.70

    def test_complexity_weight_two_words(self, scorer):
        """2-word query → weight = 0.70."""
        assert scorer._query_complexity_weight("machine learning") == 0.70

    def test_complexity_weight_three_words(self, scorer):
        """3-word query → weight = 0.85."""
        assert scorer._query_complexity_weight("what is AI") == 0.85

    def test_complexity_weight_five_words(self, scorer):
        """5-word query → weight = 0.95."""
        assert scorer._query_complexity_weight("what is deep learning anyway") == 0.95

    def test_complexity_weight_long_query(self, scorer):
        """10-word query → weight = 1.0."""
        q = "how does a transformer model learn from training data exactly"
        assert scorer._query_complexity_weight(q) == 1.00

    def test_complexity_weight_short_query(self, scorer):
        """1-word query returns the minimum weight 0.70."""
        assert scorer._query_complexity_weight("AI") == 0.70


# ═══════════════════════════════════════════════════════════════════════════
# FIX 4 — _calibrate_score (sigmoid)
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibrateScore:
    @pytest.fixture
    def scorer(self):
        from services.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        return svc

    def test_calibrate_score_midpoint(self, scorer):
        """raw = 0.60 → sigmoid midpoint → calibrated ≈ 0.500."""
        result = scorer._calibrate_score(0.60)
        assert result == pytest.approx(0.500, abs=0.002)

    def test_calibrate_score_high(self, scorer):
        """raw = 0.90 → calibrated > 0.85."""
        result = scorer._calibrate_score(0.90)
        assert result > 0.85

    def test_calibrate_score_low(self, scorer):
        """raw = 0.35 → calibrated < 0.15 (well below midpoint)."""
        result = scorer._calibrate_score(0.35)
        assert result < 0.15

    def test_calibrate_score_returns_float_in_0_1(self, scorer):
        for raw in [0.0, 0.35, 0.60, 0.82, 1.0]:
            r = scorer._calibrate_score(raw)
            assert 0.0 <= r <= 1.0, f"Out of range for raw={raw}: {r}"

    def test_calibrate_score_is_monotone(self, scorer):
        """Higher raw score must yield higher calibrated score."""
        raws = [0.35, 0.50, 0.60, 0.72, 0.82, 0.90]
        calibrated = [scorer._calibrate_score(r) for r in raws]
        assert calibrated == sorted(calibrated)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — Strict grounding prompt in _build_messages
# ═══════════════════════════════════════════════════════════════════════════

class TestStrictGroundingPrompt:
    @pytest.fixture
    def llama(self):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        svc._GROQ_MODEL = "llama-3.3-70b-versatile"
        return svc

    def test_strict_grounding_prompt_contains_only_keyword(self, llama):
        """System prompt must contain 'ONLY' or 'only' to enforce grounding."""
        messages = llama._build_messages(
            "What is photosynthesis?",
            context="Plants use sunlight to produce energy via photosynthesis.",
        )
        system_content = messages[0]["content"]
        assert "only" in system_content.lower() or "ONLY" in system_content

    def test_strict_grounding_prompt_contains_not_found_instruction(self, llama):
        """System prompt must reference the not-found message so the LLM knows to use it."""
        messages = llama._build_messages("Any question?", context="Some context.")
        system_content = messages[0]["content"]
        assert "cannot find" in system_content.lower() or "not present" in system_content.lower()

    def test_no_context_returns_not_found_without_api_call(self, llama):
        """generate_answer() with context=None returns NOT_FOUND_MSG immediately."""
        llama._client = MagicMock()
        result = llama.generate_answer("Anything?", context=None)
        assert result == llama.NOT_FOUND_MSG
        llama._client.chat.completions.create.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 5 — search_with_rerank fallback when reranker=None
# ═══════════════════════════════════════════════════════════════════════════

class TestRerankerFallback:
    @pytest.fixture
    def chroma(self):
        from services.chroma_service import ChromaService
        svc = ChromaService.__new__(ChromaService)
        svc.reranker = None
        return svc

    def test_reranker_fallback_when_none(self, chroma):
        """search_with_rerank with reranker=None falls back to search() results."""
        mock_passages = [
            {"text": "Passage A", "similarity_score": 0.85,
             "source": "a.pdf", "metadata": {}, "page": 0},
            {"text": "Passage B", "similarity_score": 0.72,
             "source": "b.pdf", "metadata": {}, "page": 0},
        ]
        with patch.object(chroma, "search", return_value=mock_passages):
            results = chroma.search_with_rerank("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "Passage A"

    def test_reranker_fallback_respects_top_k(self, chroma):
        """When reranker=None, only top_k results are returned from search()."""
        passages = [
            {"text": f"Passage {i}", "similarity_score": 1.0 - i * 0.1,
             "source": "doc.pdf", "metadata": {}, "page": 0}
            for i in range(10)
        ]
        with patch.object(chroma, "search", return_value=passages):
            results = chroma.search_with_rerank("query", top_k=3)

        assert len(results) == 3

    def test_reranker_fallback_empty_passages(self, chroma):
        """search_with_rerank returns [] gracefully when search() returns []."""
        with patch.object(chroma, "search", return_value=[]):
            results = chroma.search_with_rerank("query", top_k=5)
        assert results == []
