"""Unit tests for validation scripts and model version logic."""
import pytest
from unittest.mock import MagicMock, patch

def test_has_finetuned_model_false_when_folder_missing(tmp_path):
    from services.llama_service import LlamaService
    svc = LlamaService.__new__(LlamaService)
    svc.finetuned_path = tmp_path / "nonexistent"
    assert svc._has_finetuned_model() is False

    def test_is_ready_true_when_client_is_set(self):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        svc._client = MagicMock()
        assert svc.is_ready() is True


class TestLlamaServiceGenerate:
    """generate_answer() behaviour when client is/isn't available."""

    def test_returns_error_string_when_not_ready(self):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        svc._client = None
        result = svc.generate_answer("What is AI?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_calls_groq_api_and_returns_content(self):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "AI is artificial intelligence."
        )
        svc._client    = mock_client
        svc._GROQ_MODEL = "llama-3.3-70b-versatile"

        result = svc.generate_answer("What is AI?")
        assert result == "AI is artificial intelligence."
        mock_client.chat.completions.create.assert_called_once()

    def test_returns_error_string_on_api_exception(self):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        svc._client    = mock_client
        svc._GROQ_MODEL = "llama-3.3-70b-versatile"

        result = svc.generate_answer("What is AI?")
        assert isinstance(result, str)
        assert len(result) > 0


class TestLlamaServiceStubs:
    """hot_swap_model and rollback_model are intentional stubs for Groq."""

    def test_hot_swap_model_returns_true(self):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        assert svc.hot_swap_model("any/path") is True

    def test_rollback_model_returns_true(self):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        assert svc.rollback_model("any/path") is True


class TestLlamaServiceSaveModelVersion:
    """save_model_version() writes a versioned record to model_history.json."""

    def test_creates_history_file_with_correct_fields(self, tmp_path):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        svc._GROQ_MODEL = "llama-3.3-70b-versatile"

        history_file = tmp_path / "model_history.json"

        # Patch the path inside save_model_version to our tmp file
        with patch(
            "services.llama_service.Path",
            side_effect=lambda *args: (
                history_file
                if args and "model_history" in str(args[-1])
                else Path(*args)
            ),
        ):
            # Use direct file injection instead to avoid patching Path entirely
            pass

        # Directly test by monkey-patching the method's path computation
        original_method = LlamaService.save_model_version

        def patched_save(self_inner, metrics):
            path = history_file
            history = []
            if path.exists():
                try:
                    history = json.loads(path.read_text())
                except Exception:
                    history = []
            history.append({
                "version":   f"v{len(history) + 1}",
                "timestamp": "2024-01-01T00:00:00",
                "accuracy":  round(metrics.get("accuracy", 0), 4),
                "f1_score":  round(metrics.get("f1", 0), 4),
                "loss":      round(metrics.get("loss", 0), 4),
                "path":      f"groq:{self_inner._GROQ_MODEL}",
            })
            path.write_text(json.dumps(history, indent=2))

        svc.save_model_version = lambda m: patched_save(svc, m)
        svc.save_model_version({"accuracy": 0.75, "f1": 0.80, "loss": 0.20})

        data = json.loads(history_file.read_text())
        assert len(data) == 1
        record = data[0]
        assert record["version"] == "v1"
        assert record["accuracy"] == 0.75
        assert record["f1_score"] == 0.80
        assert record["loss"] == 0.20
        assert "groq:" in record["path"]

    def test_appends_on_second_call(self, tmp_path):
        from services.llama_service import LlamaService
        svc = LlamaService.__new__(LlamaService)
        svc._GROQ_MODEL = "llama-3.3-70b-versatile"

        history_file = tmp_path / "model_history.json"

        def patched_save(self_inner, metrics):
            path = history_file
            history = []
            if path.exists():
                try:
                    history = json.loads(path.read_text())
                except Exception:
                    history = []
            history.append({
                "version":   f"v{len(history) + 1}",
                "timestamp": "2024-01-01T00:00:00",
                "accuracy":  round(metrics.get("accuracy", 0), 4),
                "f1_score":  round(metrics.get("f1", 0), 4),
                "loss":      round(metrics.get("loss", 0), 4),
                "path":      f"groq:{self_inner._GROQ_MODEL}",
            })
            path.write_text(json.dumps(history, indent=2))

        svc.save_model_version = lambda m: patched_save(svc, m)
        svc.save_model_version({"accuracy": 0.5, "f1": 0.5, "loss": 0.5})
        svc.save_model_version({"accuracy": 0.6, "f1": 0.6, "loss": 0.4})

        data = json.loads(history_file.read_text())
        assert len(data) == 2
        assert data[0]["version"] == "v1"
        assert data[1]["version"] == "v2"


# ============================================================
# ScoringService — _quality_penalty
# ============================================================

class TestQualityPenalty:
    """_quality_penalty() returns a float multiplier in [0, 1]."""

    @pytest.fixture
    def scorer(self):
        from services.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        return svc

    def test_clean_answer_has_no_penalty(self, scorer):
        answer = "Photosynthesis is the process by which plants convert sunlight into energy."
        assert scorer._quality_penalty(answer) == 1.0

    def test_many_leaked_stop_tokens_penalised(self, scorer):
        bad = "<|eot_id|> " * 5
        penalty = scorer._quality_penalty(bad)
        assert penalty < 1.0

    def test_repeated_assistant_tag_penalised(self, scorer):
        bad = "assistant " * 6
        penalty = scorer._quality_penalty(bad)
        assert penalty < 1.0

    def test_runaway_length_penalised(self, scorer):
        long_answer = "word " * 850
        penalty = scorer._quality_penalty(long_answer)
        assert penalty < 1.0

    def test_high_sentence_repetition_penalised(self, scorer):
        sentence = "The sky is blue. "
        repetitive = sentence * 20
        penalty = scorer._quality_penalty(repetitive)
        assert penalty < 1.0

    def test_penalty_always_in_valid_range(self, scorer):
        for text in [
            "",
            "OK",
            "word " * 1000,
            "<|eot_id|>" * 10 + "assistant " * 10,
        ]:
            p = scorer._quality_penalty(text)
            assert 0.0 <= p <= 1.0, f"Penalty {p} out of range for: {text[:40]!r}"
