"""
Llama Language Model Service — Groq API
~1-2 second inference via Groq's hosted LLaMA-3.2-3B
No local model loading, no RAM overhead, no OOMKills.
"""

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

from groq import Groq
from huggingface_hub import login

from core.config import settings

logger = logging.getLogger(__name__)

 
class LlamaService:
    """Uses Groq API for fast LLaMA inference (~1-2 seconds per query)."""

    _GROQ_MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        self._client: Optional[Groq] = None
        self.model_name = settings.LLAMA_MODEL_NAME
        self.finetuned_path = Path("data/finetuned_model")
        logger.info("LlamaService (Groq API) — initialising")
        self._initialize()

    def _initialize(self) -> None:
        try:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                logger.error("GROQ_API_KEY environment variable not set!")
                return

            self._client = Groq(api_key=api_key)
            logger.info(f"Groq client initialised — model: {self._GROQ_MODEL}")

            if settings.HUGGINGFACE_TOKEN:
                try:
                    login(token=settings.HUGGINGFACE_TOKEN)
                    logger.info("HuggingFace login successful")
                except Exception:
                    logger.warning("HuggingFace login failed — not needed for Groq")

        except Exception:
            logger.exception("Failed to initialise LlamaService (Groq)")
            self._client = None

    # Canonical not-found message returned whenever grounding fails.
    NOT_FOUND_MSG = (
        "I cannot find information about this in the provided documents."
    )

    def _build_messages(self, question: str, context: Optional[str]) -> list:
        system = (
            "You are a precise AI assistant grounded strictly in the provided documents. "
            "Answer the user's question using ONLY the information explicitly stated in the "
            "context below. Do NOT use your training knowledge, make inferences, or fill in "
            "gaps beyond what the context says. "
            "If the answer is not present in the context, respond with exactly: "
            f'"{self.NOT_FOUND_MSG}"'
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        if context is None:
            logger.info(
                "[LLM] No context provided — returning not-found message without API call"
            )
            return self.NOT_FOUND_MSG

        if not self.is_ready():
            logger.warning("Groq client not initialised — check GROQ_API_KEY")
            return "Model not available. Please check API configuration."

        messages = self._build_messages(question, context)

        try:
            response = self._client.chat.completions.create(
                model       = self._GROQ_MODEL,
                messages    = messages,
                max_tokens  = 512,
                temperature = 0.0,
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer (Groq) — {len(answer)} chars")
            return answer

        except Exception:
            logger.exception("Groq API error in generate_answer")
            return "Error generating answer."

    def hot_swap_model(self, new_model_path: str) -> bool:
        logger.info(f"[HotSwap] Groq uses hosted model, ignoring path '{new_model_path}'")
        return True

    def rollback_model(self, backup_path: str) -> bool:
        logger.info(f"[Rollback] Groq uses hosted model, ignoring path '{backup_path}'")
        return True

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
        history.append({
            "version":   f"v{len(history) + 1}",
            "timestamp": datetime.now().isoformat(),
            "accuracy":  round(metrics.get("accuracy", 0), 4),
            "f1_score":  round(metrics.get("f1", 0), 4),
            "loss":      round(metrics.get("loss", 0), 4),
            "path":      f"groq:{self._GROQ_MODEL}",
        })
        history_path.write_text(json.dumps(history, indent=2))
        logger.info(f"[ModelHistory] Saved version v{len(history)}")

    def retrain(self, gold_data, hard_data, status_callback=None) -> None:
        def _cb(p, m):
            logger.info(f"[Retrain {p:3d}%] {m}")
            if status_callback: status_callback(p, m)
        _cb(0,   "Groq API does not support fine-tuning.")
        _cb(100, "Retrain skipped.")

    def is_ready(self) -> bool:
        return self._client is not None