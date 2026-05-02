"""
Llama Language Model Service — Groq API
~1-2 second inference via Groq's hosted LLaMA-3.3-70B
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from groq import Groq
from huggingface_hub import login

from core.config import settings

logger = logging.getLogger(__name__)


class LlamaService:
    """Uses Groq API for fast LLaMA inference (~1-2 seconds per query)."""

    _GROQ_MODEL = "llama-3.3-70b-versatile"

    NOT_FOUND_MSG = (
        "I cannot find relevant information about this topic in the knowledge base."
    )

    def __init__(self):
        self._client: Optional[Groq] = None
        self.model_name     = settings.LLAMA_MODEL_NAME
        self.finetuned_path = Path("data/finetuned_model")
        logger.info("LlamaService (Groq API) — initialising")
        self._initialize()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        try:
            # FIX: use settings (pydantic-settings reads .env) instead of
            # os.environ.get() which bypasses .env loading entirely.
            api_key = settings.GROQ_API_KEY
            if not api_key:
                logger.error("GROQ_API_KEY is not set — check your .env file!")
                return

            self._client = Groq(api_key=api_key)
            logger.info("Groq client initialised — model: %s", self._GROQ_MODEL)

            if settings.HUGGINGFACE_TOKEN:
                try:
                    login(token=settings.HUGGINGFACE_TOKEN)
                    logger.info("HuggingFace login successful")
                except Exception:
                    logger.warning("HuggingFace login failed — not needed for Groq")

        except Exception:
            logger.exception("Failed to initialise LlamaService (Groq)")
            self._client = None

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_messages(self, question: str, context: str) -> list:
        """
        Build the chat messages list for the Groq API.

        WHY THE OLD PROMPT GAVE 1-2 SENTENCE ANSWERS:
        -----------------------------------------------
        Old: "Answer using ONLY information explicitly stated ... Do NOT infer"
        + temperature=0.0 → the model literally quotes 1-2 sentences and stops.
        It has no instruction to explain, elaborate, or structure the answer.

        New strategy:
        - Ask the model to explain and synthesise across passages.
        - Explicitly require a minimum level of detail.
        - Allow coherent rephrasing (not just literal quoting).
        - Keep grounding: every claim must come from the context.
        - temperature=0.1 (set in generate_answer) adds just enough fluency.
        """
        system = (
            "You are an expert AI assistant helping users understand a knowledge base. "
            "You receive numbered context passages from those documents.\n\n"
            "YOUR TASK:\n"
            "1. Read ALL provided context passages carefully.\n"
            "2. Give a COMPREHENSIVE, WELL-STRUCTURED answer — not just a single "
            "sentence. Elaborate on the concept, define terms, and explain clearly.\n"
            "3. Synthesise information across passages when relevant. "
            "You may rephrase content — you do NOT need to quote word-for-word.\n"
            "4. Every claim must be grounded in the context. Do NOT add information "
            "from your training data that is not in the passages.\n"
            "5. If the context genuinely does not contain enough information to answer, "
            f"respond with exactly:\n\"{self.NOT_FOUND_MSG}\"\n\n"
            "OUTPUT FORMAT:\n"
            "- Write in clear prose paragraphs (not bullet points unless natural).\n"
            "- Define any technical terms when they first appear.\n"
            "- Provide examples from the context where available.\n"
            "- Minimum 3 sentences; aim for a thorough explanation."
        )
        user = (
            f"CONTEXT PASSAGES:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide a detailed, thorough answer based solely on the context above."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    # ── Answer generation ──────────────────────────────────────────────────────

    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        if context is None:
            logger.info("[LLM] No context — returning not-found without API call")
            return self.NOT_FOUND_MSG

        if not self.is_ready():
            logger.warning("Groq client not initialised — check GROQ_API_KEY in .env")
            return "Model not available. Please check API configuration."

        messages = self._build_messages(question, context)

        try:
            response = self._client.chat.completions.create(
                model       = self._GROQ_MODEL,
                messages    = messages,
                # FIX: 512 tokens ≈ 380 words — one short paragraph at best.
                # 1024 tokens ≈ 750 words — enough for a proper explanation.
                max_tokens  = 728,
                # FIX: 0.0 forces the most conservative, literal output.
                # 0.1 produces fluent, coherent explanations while staying grounded.
                temperature = 0.3,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("[LLM] Generated answer — %d chars", len(answer))
            return answer

        except Exception:
            logger.exception("Groq API error in generate_answer")
            return "Error generating answer. Please try again."

    # ── Model lifecycle (Groq is hosted — these are no-ops) ───────────────────

    def hot_swap_model(self, new_model_path: str) -> bool:
        logger.info("[HotSwap] Groq is hosted — ignoring path '%s'", new_model_path)
        return True

    def rollback_model(self, backup_path: str) -> bool:
        logger.info("[Rollback] Groq is hosted — ignoring path '%s'", backup_path)
        return True

    def save_model_version(self, metrics: dict) -> None:
        history_path = (
            Path(__file__).resolve().parent.parent / "data" / "model_history.json"
        )
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
        logger.info("[ModelHistory] Saved version v%d", len(history))

    def retrain(self, gold_data, hard_data, status_callback=None) -> dict:
        """Run retraining pipeline and persist the metrics."""
        from services.retraining_service import RetrainingService

        def _cb(p: int, m: str) -> None:
            logger.info("[Retrain %3d%%] %s", p, m)
            if status_callback:
                status_callback(p, m)

        service = RetrainingService()
        metrics = service.run(
            gold_data=gold_data,
            hard_data=hard_data,
            status_callback=_cb,
        )
        self.save_model_version(metrics)
        return metrics

    def is_ready(self) -> bool:
        return self._client is not None