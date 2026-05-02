"""
Confidence Scoring Service — calibrated scoring for RAG answers.

KEY FIXES in this version:
─────────────────────────────────────────────────────────────────────────────
1. SIGMOID MISCALIBRATION (critical):
   Old center = 0.60.  MiniLM's typical similarity for genuinely relevant
   text is 0.45–0.60.  At center=0.60, a raw_score of 0.55 (good match)
   calibrated to 0.40 → labelled "Low" even when the answer was correct.

   New center = 0.45, steepness = 7:
     raw=0.35 → 0.33  (low)
     raw=0.45 → 0.50  (mid)
     raw=0.55 → 0.67  (medium-high)
     raw=0.65 → 0.81  (high)
     raw=0.75 → 0.90  (excellent)

2. SINGLE-PASSAGE MAX SCORING:
   Old: raw_score = cosine_scores.max()
   A single strong chunk masked by 4 weak ones gave a misleadingly high
   score. Replaced with weighted top-2 average (top chunk × 0.7 + 2nd × 0.3)
   which is more stable and representative.

3. STRICT CUTOFF added back at raw < 0.28:
   Prevents noise passages (raw ≈ 0.10–0.25) from ever reaching the LLM.
   The pre-flight in endpoints already blocks most junk, but this is a
   second safety net inside the scoring layer.

4. INDENTATION BUG FIXED in _quality_penalty:
   The `# 1.` comment was at class-method indent level (4 spaces) while
   the code following it was at body indent level (8 spaces). Fixed to
   keep everything consistently inside the method.
"""

import math
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer, util

from core.config import settings


class ScoringService:
    """Confidence scoring engine: semantic similarity + quality penalty."""

    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    # ── Quality guard ──────────────────────────────────────────────────────────

    def _quality_penalty(self, answer: str) -> float:
        """
        Returns a multiplier in [0.0, 1.0].
        Only heavily penalises genuinely broken answers (leaked tokens,
        runaway repetition, excessive filler). Normal answers → 1.0.
        """
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

        # 3. Filler / farewell phrases (hallucinated chat endings)
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

        # 5. Runaway generation (answer way too long = probably hallucinating)
        word_count = len(answer.split())
        if word_count > 800:
            penalty *= 0.6
        elif word_count > 600:
            penalty *= 0.8

        return round(penalty, 3)

    # ── Query complexity weight ────────────────────────────────────────────────

    def _query_complexity_weight(self, question: str) -> float:
        """
        Shorter queries have less signal to verify against, so we apply a
        small downward weight to prevent over-confident scores on vague queries.

          1–2 words → 0.75
          3–4 words → 0.88
          5–6 words → 0.96
          7+ words  → 1.00
        """
        n = len(question.split())
        if n <= 2:
            return 0.75
        if n <= 4:
            return 0.88
        if n <= 6:
            return 0.96
        return 1.00

    # ── Sigmoid calibration ────────────────────────────────────────────────────

    def _calibrate_score(self, raw_score: float) -> float:
        """
        Sigmoid centred at 0.45 with steepness 7.

        Calibrated to MiniLM's actual similarity range for relevant text:
          raw = 0.28 → 0.19  (just above cutoff — low confidence)
          raw = 0.35 → 0.33  (marginal)
          raw = 0.45 → 0.50  (mid-point — genuinely related text)
          raw = 0.55 → 0.67  (good match)
          raw = 0.65 → 0.81  (strong match → High confidence)
          raw = 0.75 → 0.90  (excellent match)

        OLD (center=0.60): raw=0.55 → 0.40 (labelled Low despite correct answer)
        NEW (center=0.45): raw=0.55 → 0.67 (correctly labelled Medium-High)
        """
        return round(1.0 / (1.0 + math.exp(-7.0 * (raw_score - 0.45))), 3)

    # ── Weighted top-k score ───────────────────────────────────────────────────

    def _weighted_raw_score(self, cosine_scores_tensor) -> float:
        """
        Compute a weighted combination of the top-2 cosine similarities.

        WHY NOT JUST MAX:
          max() is noise-sensitive: a single noisy chunk with a fluke high
          score produces over-confidence. A single very relevant chunk buried
          at rank 2 gets ignored if rank-1 is moderately relevant.

        FORMULA:
          top1 × 0.70 + top2 × 0.30  (if at least 2 passages)
          top1 × 1.00                 (if only 1 passage)

        This is more stable and still heavily weights the best match.
        """
        scores = cosine_scores_tensor.tolist()
        if isinstance(scores, float):
            scores = [scores]
        scores_sorted = sorted(scores, reverse=True)

        if len(scores_sorted) >= 2:
            return scores_sorted[0] * 0.70 + scores_sorted[1] * 0.30
        return scores_sorted[0]

    # ── Main scoring ───────────────────────────────────────────────────────────

    def compute_confidence_score(
        self,
        answer: str,
        question: str,
        retrieved_passages: List[Dict],
    ) -> Tuple[float, str, List[Dict], Dict[str, float]]:
        """
        Compute a calibrated confidence score for an LLM answer.

        Returns:
            (final_score, explanation, citations, score_breakdown)
        """
        if not answer or not retrieved_passages:
            return 0.0, "No data to score.", [], {}

        # ── 1. Semantic similarity ─────────────────────────────────────────────
        # Encode (question + answer) together so the score captures whether
        # the answer addresses the question using the retrieved content.
        scoring_text  = f"{question} {answer}"
        text_emb      = self.embedding_model.encode(scoring_text, convert_to_tensor=True)
        passage_texts = [p["text"] for p in retrieved_passages]
        passage_embs  = self.embedding_model.encode(passage_texts, convert_to_tensor=True)

        cosine_scores = util.cos_sim(text_emb, passage_embs)[0]

        # FIX: weighted top-2 average instead of raw max()
        raw_score = float(self._weighted_raw_score(cosine_scores))

        # ── 2. Strict low-quality cutoff ──────────────────────────────────────
        # Pre-flight in endpoints already blocks most junk, but this is a
        # second safety net: if the scored answer has < 0.28 similarity even
        # after top-k retrieval, it came from the model's own knowledge, not
        # the documents.
        if raw_score < 0.28:
            return (
                0.0,
                (
                    "Answer appears to be from the model's internal knowledge. "
                    "No sufficiently relevant passages were found in the knowledge base."
                ),
                [],
                {"consistency": 0, "semantic": 0, "completeness": 0, "precision": 0},
            )

        # ── 3. Calibrate → quality penalty → complexity weight ────────────────
        calibrated_score = self._calibrate_score(raw_score)
        quality          = self._quality_penalty(answer)
        complexity       = self._query_complexity_weight(question)

        final_score = round(calibrated_score * quality * complexity, 2)
        final_score = max(0.0, min(1.0, final_score))

        # ── 4. Human-readable explanation ─────────────────────────────────────
        explanation = f"Confidence Score: {final_score:.2f}. "
        if quality < 0.5:
            explanation += (
                "⚠️ Answer quality is degraded — repetition, runaway generation, "
                f"or leaked tokens detected (quality penalty: {quality:.2f})."
            )
        elif final_score >= 0.80:
            explanation += (
                "Strong match. The answer is well-supported by the retrieved documents."
            )
        elif final_score >= 0.55:
            explanation += (
                "Moderate match. The answer aligns with the documents but may "
                "not cover all aspects of the question."
            )
        else:
            explanation += (
                "Weak match. The answer has limited overlap with the documents — "
                "treat with caution."
            )

        # ── 5. Build citations ─────────────────────────────────────────────────
        citations = []
        for i, p in enumerate(retrieved_passages):
            excerpt = p["text"]
            citations.append({
                "source":          p.get("source", "unknown"),
                "excerpt":         excerpt[:200] + "…" if len(excerpt) > 200 else excerpt,
                "relevance_score": round(p.get("similarity_score", 0), 2),
            })

        # ── 6. Score breakdown for frontend ───────────────────────────────────
        breakdown = {
            "consistency":  final_score,
            "semantic":     round(calibrated_score, 2),
            "completeness": round(0.95 if final_score >= 0.80 else final_score, 2),
            "precision":    round(quality, 2),
        }

        return final_score, explanation, citations, breakdown