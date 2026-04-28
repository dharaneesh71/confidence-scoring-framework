"""
Llama Language Model Service — llama-cpp-python (GGUF quantized, CPU optimised)
~3-4x faster than HuggingFace transformers on CPU inference.
"""

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama

from core.config import settings

logger = logging.getLogger(__name__)

_GGUF_REPO    = "bartowski/Llama-3.2-3B-Instruct-GGUF"
_GGUF_FILE    = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
_GGUF_CACHE   = Path("data/gguf_models")

_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
_EOT              = "<|eot_id|>"


class LlamaService:
    _GEN_CONFIG = dict(
        max_tokens     = 200,
        stop           = [_EOT, "<|end_of_text|>"],
        echo           = False,
        repeat_penalty = 1.1,
        temperature    = 0.0,
    )

    def __init__(self):
        self.model: Optional[Llama] = None
        self.model_name = settings.LLAMA_MODEL_NAME
        logger.info("LlamaService (llama-cpp) — initialising")
        self._initialize()

    def _initialize(self) -> None:
        try:
            n_threads = int(os.environ.get("OMP_NUM_THREADS", 16))
            logger.info(f"llama-cpp using {n_threads} CPU threads")

            if settings.HUGGINGFACE_TOKEN:
                login(token=settings.HUGGINGFACE_TOKEN)
                logger.info("HuggingFace login successful")

            _GGUF_CACHE.mkdir(parents=True, exist_ok=True)
            local_path = _GGUF_CACHE / _GGUF_FILE

            if not local_path.exists():
                logger.info(f"Downloading GGUF model: {_GGUF_FILE} ...")
                hf_hub_download(
                    repo_id   = _GGUF_REPO,
                    filename  = _GGUF_FILE,
                    token     = settings.HUGGINGFACE_TOKEN,
                    local_dir = str(_GGUF_CACHE),
                )
            else:
                logger.info(f"GGUF model found in cache: {local_path}")

            self.model = Llama(
                model_path   = str(local_path),
                n_ctx        = 2048,
                n_threads    = n_threads,
                n_gpu_layers = 0,
                verbose      = False,
            )
            logger.info("LlamaService (llama-cpp) initialised successfully")

        except Exception:
            logger.exception("Failed to initialise LlamaService")
            self.model = None

    def _build_prompt(self, question: str, context: Optional[str]) -> str:
        if context:
            user_text   = f"Context:\n{context}\n\nQuestion: {question}"
            system_text = (
                "You are a precise AI assistant. Answer the question using only the "
                "given context. Be direct, factual, and concise (2-4 sentences). "
                "If the context does not contain the answer, say so briefly."
            )
        else:
            user_text   = question
            system_text = (
                "You are a precise AI assistant. Give a clear, focused answer in "
                "2-4 sentences. Be direct and concise. Avoid filler or repetition."
            )
        return (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_text}{_EOT}\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text}{_EOT}\n"
            f"{_ASSISTANT_HEADER}"
        )

    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        if not self.is_ready():
            return "Model not loaded."
        prompt = self._build_prompt(question, context)
        try:
            response = self.model(prompt, **self._GEN_CONFIG)
            answer   = response["choices"][0]["text"].strip()
            for tok in [_EOT, "<|end_of_text|>"]:
                answer = answer.replace(tok, "").strip()
            logger.info(f"Generated answer — {len(answer)} chars")
            return answer
        except Exception:
            logger.exception("Error in generate_answer")
            return "Error generating answer."

    def hot_swap_model(self, new_model_path: str) -> bool:
        old_model = self.model
        try:
            n_threads  = int(os.environ.get("OMP_NUM_THREADS", 16))
            self.model = Llama(model_path=new_model_path, n_ctx=2048, n_threads=n_threads, n_gpu_layers=0, verbose=False)
            del old_model
            return True
        except Exception:
            self.model = old_model
            return False

    def rollback_model(self, backup_path: str) -> bool:
        return self.hot_swap_model(backup_path)

    def save_model_version(self, metrics: dict) -> None:
        import json
        from datetime import datetime
        history_path = Path("data/model_history.json")
        history: list = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except Exception:
                history = []
        history.append({"version": f"v{len(history)+1}", "timestamp": datetime.now().isoformat(),
                        "accuracy": round(metrics.get("accuracy",0),4), "f1_score": round(metrics.get("f1",0),4),
                        "loss": round(metrics.get("loss",0),4), "path": _GGUF_FILE})
        history_path.write_text(json.dumps(history, indent=2))

    def retrain(self, gold_data, hard_data, status_callback=None) -> None:
        if status_callback:
            status_callback(0,  "llama-cpp does not support fine-tuning.")
            status_callback(100,"Retrain skipped.")

    def is_ready(self) -> bool:
        return self.model is not None
