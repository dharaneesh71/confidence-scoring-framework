"""
Confidence Scoring Service — calibrated scoring for RAG answers.

PRODUCTION VERSION — TF-IDF scoring (no PyTorch):
─────────────────────────────────────────────────────────────────────────────
1. TFIDF SCORING: Replaced SentenceTransformer + util.cos_sim with
   sklearn TfidfVectorizer + cosine_similarity. No PyTorch dependency.

2. SIGMOID CALIBRATION: center=0.45, steepness=7 for TF-IDF score range.

3. WEIGHTED TOP-2: top × 0.7 + 2nd × 0.3 (more stable than max).

4. STRICT CUTOFF at raw < 0.10 (adjusted for TF-IDF range vs MiniLM).

5. QUALITY PENALTY: Detects leaked tokens, repetition, runaway generation.
"""

import math
import numpy as np
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from core.config import settings


class ScoringService:
    """Confidence scoring engine: TF-IDF similarity + quality penalty."""

    def __init__(self) -> None:
        # No model loading — TF-IDF computed fresh per request
        # This means zero startup time, zero memory for ML models
        pass

    # ── TF-IDF similarity ──────────────────────────────────────────────────────

    def _compute_similarities(
        self,
        query_text: str,
        passage_texts: List[str],
    ) -> List[float]:
        """
        Compute TF-IDF cosine similarities between query and passages.
        Fits a fresh vectorizer on all texts together per request.
        """
        if not passage_texts:
            return []

        try:
            all_texts = [query_text] + passage_texts
            vectorizer = TfidfVectorizer(
                max_features=384,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
            )
            tfidf_matrix = vectorizer.fit_transform(all_texts).toarray()
            query_vec    = tfidf_matrix[0:1]
            passage_vecs = tfidf_matrix[1:]
            similarities = sklearn_cosine(query_vec, passage_vecs)[0].tolist()
            return similarities
        except Exception:
            return [0.0] * len(passage_texts)

    # ── Quality guard ──────────────────────────────────────────────────────────

    def _quality_penalty(self, answer: str) -> float:
        """Returns a multiplier in [0.0, 1.0]. Normal answers → 1.0."""
        penalty = 1.0

        # 1. Leaked stop tokens
        bad_tokens = ["<|start_header_id|>", "<|eot_id|>", "<|end_of_text|>"]
        leaked = sum(answer.count(t) for t in bad_tokens)
        if leaked > 3:
            penalty *= 0.6
        elif leaked > 0:
            penalty *= 0.85

        # 2. Repeated "assistant" tag
        assistant_count = answer.lower().count("assistant")
        if assistant_count > 5:
            penalty *= 0.5
        elif assistant_count > 2:
            penalty *= 0.75

        # 3. Filler / farewell phrases
        filler_phrases = [
            "goodbye", "bye!", "see you", "take care",
            "peace out", "farewell", "later!", "outta here",
        ]
        filler_count = sum(answer.lower().count(p) for p in filler_phrases)
        if filler_count > 5:
            penalty *= 0.6
        elif filler_count > 2:
            penalty *= 0.8

        # 4. Sentence-level repetition
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        if sentences:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.3:
                penalty *= 0.6
            elif unique_ratio < 0.5:
                penalty *= 0.8

        # 5. Runaway generation
        word_count = len(answer.split())
        if word_count > 800:
            penalty *= 0.6
        elif word_count > 600:
            penalty *= 0.8

        return round(penalty, 3)

    # ── Query complexity weight ────────────────────────────────────────────────

    def _query_complexity_weight(self, question: str) -> float:
        n = len(question.split())
        if n <= 2:  return 0.75
        if n <= 4:  return 0.88
        if n <= 6:  return 0.96
        return 1.00

    # ── Sigmoid calibration ────────────────────────────────────────────────────

    def _calibrate_score(self, raw_score: float) -> float:
        """
        Sigmoid centred at 0.30 with steepness 8 (calibrated for TF-IDF range).
        TF-IDF scores are typically lower than MiniLM scores.
          raw=0.10 → 0.18   raw=0.30 → 0.50
          raw=0.20 → 0.31   raw=0.40 → 0.69
          raw=0.50 → 0.85   raw=0.60 → 0.93
        """
        return round(1.0 / (1.0 + math.exp(-8.0 * (raw_score - 0.30))), 3)

    # ── Weighted top-k score ───────────────────────────────────────────────────

    def _weighted_raw_score(self, cosine_scores: List[float]) -> float:
        """Weighted top-2: top1 × 0.70 + top2 × 0.30."""
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
        """
        Compute calibrated confidence score for an LLM answer.

        Returns:
            (final_score, explanation, citations, score_breakdown)
        """
        if not answer or not retrieved_passages:
            return 0.0, "No data to score.", [], {}

        # ── 1. TF-IDF similarity ───────────────────────────────────────────
        scoring_text  = f"{question} {answer}"
        passage_texts = [p["text"] for p in retrieved_passages]

        cosine_scores = self._compute_similarities(scoring_text, passage_texts)
        raw_score     = float(self._weighted_raw_score(cosine_scores))

        # ── 2. Low-quality cutoff (TF-IDF threshold = 0.10) ───────────────
        if raw_score < 0.10:
            return (
                0.0,
                (
                    "Answer appears to be from the model's internal knowledge. "
                    "No sufficiently relevant passages found in knowledge base."
                ),
                [],
                {"consistency": 0, "semantic": 0, "completeness": 0, "precision": 0},
            )

        # ── 3. Calibrate → quality penalty → complexity weight ────────────
        calibrated_score = self._calibrate_score(raw_score)
        quality          = self._quality_penalty(answer)
        complexity       = self._query_complexity_weight(question)

        final_score = round(calibrated_score * quality * complexity, 2)
        final_score = max(0.0, min(1.0, final_score))

        # ── 4. Explanation ─────────────────────────────────────────────────
        explanation = f"Confidence Score: {final_score:.2f}. "
        if quality < 0.5:
            explanation += (
                f"⚠️ Answer quality degraded (quality penalty: {quality:.2f})."
            )
        elif final_score >= 0.80:
            explanation += "Strong match — well-supported by retrieved documents."
        elif final_score >= 0.55:
            explanation += "Moderate match — aligns with documents but may miss some aspects."
        else:
            explanation += "Weak match — limited overlap with documents. Treat with caution."

        # ── 5. Citations ───────────────────────────────────────────────────
        citations = []
        for p in retrieved_passages:
            excerpt = p["text"]
            citations.append({
                "source":          p.get("source", "unknown"),
                "excerpt":         excerpt[:200] + "…" if len(excerpt) > 200 else excerpt,
                "relevance_score": round(p.get("similarity_score", 0), 2),
            })

        # ── 6. Breakdown ───────────────────────────────────────────────────
        breakdown = {
            "consistency":  final_score,
            "semantic":     round(calibrated_score, 2),
            "completeness": round(0.95 if final_score >= 0.80 else final_score, 2),
            "precision":    round(quality, 2),
        }

        return final_score, explanation, citations, breakdown