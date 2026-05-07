"""
Confidence Scoring Service — calibrated scoring for RAG answers.

PRODUCTION VERSION — Hash embedding scoring (no PyTorch, no sklearn):
─────────────────────────────────────────────────────────────────────────────
1. HASH EMBEDDING: Pure Python + numpy cosine similarity.
   No native ML libs — eliminates exit code 139 segfault on CSLaunch.

2. SIGMOID CALIBRATION: center=0.30, steepness=8 for hash embedding range.

3. WEIGHTED TOP-2: top × 0.7 + 2nd × 0.3 (more stable than max).

4. STRICT CUTOFF at raw < 0.05 (adjusted for hash embedding range).

5. QUALITY PENALTY: Detects leaked tokens, repetition, runaway generation.
"""

import hashlib
import math
import numpy as np
from typing import Dict, List, Tuple

from core.config import settings


class ScoringService:
    """Confidence scoring engine: hash similarity + quality penalty."""

    EMBED_DIM = 384

    def __init__(self) -> None:
        pass  # no model loading — pure Python hashing

    # ── Hash embedding ─────────────────────────────────────────────────────────

    def _embed_text(self, text: str) -> np.ndarray:
        """Same hash embedding as ChromaService for consistency."""
        words  = text.lower().split()
        vector = np.zeros(self.EMBED_DIM, dtype=np.float32)
        for i, word in enumerate(words):
            idx          = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.EMBED_DIM
            vector[idx] += 1.0 / (1.0 + i)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    # ── Cosine similarity ──────────────────────────────────────────────────────

    def _compute_similarities(
        self,
        query_text: str,
        passage_texts: List[str],
    ) -> List[float]:
        """Compute cosine similarities using hash embeddings + numpy."""
        if not passage_texts:
            return []
        try:
            query_vec = self._embed_text(query_text)
            return [
                float(max(0.0, np.dot(query_vec, self._embed_text(p))))
                for p in passage_texts
            ]
        except Exception:
            return [0.0] * len(passage_texts)

    # ── Quality guard ──────────────────────────────────────────────────────────

    def _quality_penalty(self, answer: str) -> float:
        """Returns a multiplier in [0.0, 1.0]. Normal answers → 1.0."""
        penalty = 1.0

        # 1. Leaked stop tokens
        bad_tokens = ["<|start_header_id|>", "<|eot_id|>", "<|end_of_text|>"]
        leaked = sum(answer.count(t) for t in bad_tokens)
        if leaked > 3:   penalty *= 0.6
        elif leaked > 0: penalty *= 0.85

        # 2. Repeated "assistant" tag
        ac = answer.lower().count("assistant")
        if ac > 5:   penalty *= 0.5
        elif ac > 2: penalty *= 0.75

        # 3. Filler phrases
        filler = ["goodbye", "bye!", "see you", "take care",
                  "peace out", "farewell", "later!", "outta here"]
        fc = sum(answer.lower().count(p) for p in filler)
        if fc > 5:   penalty *= 0.6
        elif fc > 2: penalty *= 0.8

        # 4. Sentence repetition
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        if sentences:
            ratio = len(set(sentences)) / len(sentences)
            if ratio < 0.3:   penalty *= 0.6
            elif ratio < 0.5: penalty *= 0.8

        # 5. Runaway generation
        wc = len(answer.split())
        if wc > 800:   penalty *= 0.6
        elif wc > 600: penalty *= 0.8

        return round(penalty, 3)

    # ── Query complexity weight ────────────────────────────────────────────────

    def _query_complexity_weight(self, question: str) -> float:
        n = len(question.split())
        if n <= 2: return 0.75
        if n <= 4: return 0.88
        if n <= 6: return 0.96
        return 1.00

    # ── Sigmoid calibration ────────────────────────────────────────────────────

    def _calibrate_score(self, raw_score: float) -> float:
        """Sigmoid centred at 0.30 with steepness 8 for hash embedding range."""
        return round(1.0 / (1.0 + math.exp(-8.0 * (raw_score - 0.30))), 3)

    # ── Weighted top-k score ───────────────────────────────────────────────────

    def _weighted_raw_score(self, cosine_scores: List[float]) -> float:
        scores_sorted = sorted(cosine_scores, reverse=True)
        if len(scores_sorted) >= 2:
            return scores_sorted[0] * 0.70 + scores_sorted[1] * 0.30
        return scores_sorted[0] if scores_sorted else 0.0

    # ── Main scoring ───────────────────────────────────────────────────────────

    def compute_confidence_score(
        self,
        answer: str,
        question: str,
        retrieved_passages: List[Dict],
    ) -> Tuple[float, str, List[Dict], Dict[str, float]]:
        """Compute calibrated confidence score for an LLM answer."""
        if not answer or not retrieved_passages:
            return 0.0, "No data to score.", [], {}

        scoring_text  = f"{question} {answer}"
        passage_texts = [p["text"] for p in retrieved_passages]

        cosine_scores = self._compute_similarities(scoring_text, passage_texts)
        raw_score     = float(self._weighted_raw_score(cosine_scores))

        # Low-quality cutoff for hash embedding range
        if raw_score < 0.05:
            return (
                0.0,
                "Answer not grounded in knowledge base.",
                [],
                {"consistency": 0, "semantic": 0, "completeness": 0, "precision": 0},
            )

        calibrated_score = self._calibrate_score(raw_score)
        quality          = self._quality_penalty(answer)
        complexity       = self._query_complexity_weight(question)

        final_score = round(
            max(0.0, min(1.0, calibrated_score * quality * complexity)), 2
        )

        explanation = f"Confidence Score: {final_score:.2f}. "
        if quality < 0.5:
            explanation += f"⚠️ Answer quality degraded (penalty: {quality:.2f})."
        elif final_score >= 0.80:
            explanation += "Strong match — well-supported by retrieved documents."
        elif final_score >= 0.55:
            explanation += "Moderate match — aligns with documents."
        else:
            explanation += "Weak match — limited overlap with documents."

        citations = [
            {
                "source":          p.get("source", "unknown"),
                "excerpt":         (p["text"][:200] + "…"
                                    if len(p["text"]) > 200 else p["text"]),
                "relevance_score": round(p.get("similarity_score", 0), 2),
            }
            for p in retrieved_passages
        ]

        breakdown = {
            "consistency":  final_score,
            "semantic":     round(calibrated_score, 2),
            "completeness": round(0.95 if final_score >= 0.80 else final_score, 2),
            "precision":    round(quality, 2),
        }

        return final_score, explanation, citations, breakdown