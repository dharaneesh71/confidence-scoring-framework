"""Unit tests for validation scripts and model version logic."""
import pytest
from unittest.mock import MagicMock, patch

def test_has_finetuned_model_false_when_folder_missing(tmp_path):
    from services.llama_service import LlamaService
    svc = LlamaService.__new__(LlamaService)
    svc.finetuned_path = tmp_path / "nonexistent"
    assert svc._has_finetuned_model() is False

def test_has_finetuned_model_false_when_folder_empty(tmp_path):
    from services.llama_service import LlamaService
    svc = LlamaService.__new__(LlamaService)
    svc.finetuned_path = tmp_path / "empty_model"
    svc.finetuned_path.mkdir()
    assert svc._has_finetuned_model() is False

def test_has_finetuned_model_true_when_safetensors_present(tmp_path):
    from services.llama_service import LlamaService
    svc = LlamaService.__new__(LlamaService)
    svc.finetuned_path = tmp_path / "model"
    svc.finetuned_path.mkdir()
    (svc.finetuned_path / "model.safetensors").touch()
    assert svc._has_finetuned_model() is True

def test_has_finetuned_model_true_when_config_present(tmp_path):
    from services.llama_service import LlamaService
    svc = LlamaService.__new__(LlamaService)
    svc.finetuned_path = tmp_path / "model"
    svc.finetuned_path.mkdir()
    (svc.finetuned_path / "config.json").touch()
    assert svc._has_finetuned_model() is True

def test_hot_swap_restores_on_failure():
    from services.llama_service import LlamaService
    svc = LlamaService.__new__(LlamaService)
    svc.device    = "cpu"
    svc.model     = MagicMock()
    svc.tokenizer = MagicMock()
    svc.pipeline  = MagicMock()
    original_model = svc.model
    with patch("services.llama_service.AutoTokenizer.from_pretrained",
               side_effect=Exception("Download failed")):
        result = svc.hot_swap_model("bad/path")
    assert result is False
    assert svc.model is original_model