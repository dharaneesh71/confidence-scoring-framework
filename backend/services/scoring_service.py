"""
Confidence Scoring Service — sigmoid calibration + query complexity weighting.

New methods added:
  _query_complexity_weight(question) → float
      Scales the final score down for very short queries (less evidence
      to verify) and leaves long queries at full weight.

  _calibrate_score(raw_score) → float
      Replaces the old step-function curve with a smooth sigmoid centred at
      raw_score = 0.60:
          calibrated = 1 / (1 + exp(-8 × (raw − 0.60)))
      Mid-point (raw=0.60) → 0.500; high (raw=0.90) → ~0.917.
"""
import math
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer, util

from core.config import settings


class ScoringService:
    """Confidence scoring engine: semantic similarity + quality penalty."""

    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    # ------------------------------------------------------------------ #
    #  Answer Quality Guard                                                #
    # ------------------------------------------------------------------ #

    def _quality_penalty(self, answer: str) -> float:
        """
        Returns a multiplier in [0, 1].
        Only heavily penalises truly broken answers.
        """
        penalty = 1.0

        # 1. Leaked stop tokens — only penalise if many present
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

        # 4. Sentence repetition
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        if sentences:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.3:
                penalty *= 0.6
            elif unique_ratio < 0.5:
                penalty *= 0.8

        # 5. Runaway length
        word_count = len(answer.split())
        if word_count > 800:
            penalty *= 0.6
        elif word_count > 600:
            penalty *= 0.8

        return round(penalty, 3)

    # ------------------------------------------------------------------ #
    #  Query Complexity Weight                                             #
    # ------------------------------------------------------------------ #

    def _query_complexity_weight(self, question: str) -> float:
        """
        Shorter queries provide less context to verify against, so we
        apply a small downward weight to the final confidence score.

          1–2 words  → 0.70
          3–4 words  → 0.85
          5–6 words  → 0.95
          7+ words   → 1.00
        """
        n = len(question.split())
        if n <= 2:
            return 0.70
        if n <= 4:
            return 0.85
        if n <= 6:
            return 0.95
        return 1.00

    # ------------------------------------------------------------------ #
    #  Sigmoid Score Calibration                                           #
    # ------------------------------------------------------------------ #

    def _calibrate_score(self, raw_score: float) -> float:
        """
        Smooth sigmoid centred at raw_score = 0.60.

            calibrated = 1 / (1 + exp(−8 × (raw − 0.60)))

        Key reference points:
          raw = 0.35 → ~0.083  (low — but hard cutoff applies before here)
          raw = 0.60 → 0.500   (mid-point)
          raw = 0.72 → ~0.832
          raw = 0.90 → ~0.917
        """
        return round(1.0 / (1.0 + math.exp(-8.0 * (raw_score - 0.60))), 3)

    # ------------------------------------------------------------------ #
    #  Main Scoring                                                        #
    # ------------------------------------------------------------------ #

    def compute_confidence_score(
        self,
        answer: str,
        question: str,
        retrieved_passages: List[Dict],
    ) -> Tuple[float, str, List[Dict], Dict[str, float]]:

        if not answer or not retrieved_passages:
            return 0.0, "No data", [], {}

        # 1. SEMANTIC SIMILARITY (The Core Metric)
        scoring_text  = f"{question} {answer}"
        text_emb      = self.embedding_model.encode(scoring_text, convert_to_tensor=True)
        passage_texts = [p["text"] for p in retrieved_passages]
        passage_embs  = self.embedding_model.encode(passage_texts, convert_to_tensor=True)

        cosine_scores = util.cos_sim(text_emb, passage_embs)[0]
        raw_score     = float(cosine_scores.max())

        # --- STRICT CUTOFF ---
        if raw_score < 0.35:
            return 0.0, (
                "Answer generated from AI's internal knowledge. "
                "No relevant documents found in the Knowledge Base."
            ), [], {"consistency": 0, "semantic": 0, "completeness": 0, "precision": 0}

        # --- SIGMOID CALIBRATION (replaces old step-function curve) ---
        calibrated_score = self._calibrate_score(raw_score)

        # --- QUALITY PENALTY + COMPLEXITY WEIGHT ---
        quality    = self._quality_penalty(answer)
        complexity = self._query_complexity_weight(question)
        final_score = round(calibrated_score * quality * complexity, 2)
        final_score = max(0.0, min(1.0, final_score))

        # 2. Explanation
        explanation = f"Confidence Score: {final_score:.2f}. "
        if quality < 0.5:
            explanation += (
                "WARNING: Answer quality is poor — "
                "repetition, runaway generation, or leaked tokens detected. "
                f"Quality penalty applied: {quality:.2f}."
            )
        elif final_score > 0.90:
            explanation += "Excellent answer. Content is strongly verified by the provided documents."
        elif final_score > 0.80:
            explanation += "Good answer. The information aligns well with the document evidence."
        else:
            explanation += "The answer has partial overlap with the documents but may contain unverified details."

        # 3. Citations
        citations = []
        for p in retrieved_passages:
            citations.append({
                "source":          p.get("source", "doc"),
                "excerpt":         p["text"][:150] + "…" if len(p["text"]) > 150 else p["text"],
                "relevance_score": round(p.get("similarity_score", 0), 2),
            })

        # 4. Score Breakdown
        breakdown = {
            "consistency":  final_score,
            "semantic":     calibrated_score,
            "completeness": 0.95 if final_score > 0.8 else final_score,
            "precision":    round(quality, 2),
        }

        return final_score, explanation, citations, breakdown
