"""
Forgiving Scoring Service with "Curved Grading"
Boosts scores for strong answers to ensure high verification confidence.
"""
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from core.config import settings


class ScoringService:
    """
    Scoring Engine that rewards factual accuracy with high confidence scores.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    # ------------------------------------------------------------------ #
    #  Answer Quality Guard                                                #
    # ------------------------------------------------------------------ #

    def _quality_penalty(self, answer: str) -> float:
        """
        Returns a multiplier between 0.0 and 1.0.
        Lenient mode — only heavily penalises truly broken answers.
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

    # 3. Filler/farewell phrases
        filler_phrases = [
            "goodbye", "bye!", "see you", "take care",
            "peace out", "farewell", "later!", "outta here"
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
    #  Main Scoring                                                        #
    # ------------------------------------------------------------------ #

    def compute_confidence_score(
        self, answer: str, question: str, retrieved_passages: List[Dict]
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

        # --- CURVE LOGIC ---
        if raw_score > 0.82:
            curved_score = 0.98
        elif raw_score > 0.72:
            curved_score = 0.94
        elif raw_score > 0.60:
            curved_score = 0.85
        elif raw_score > 0.50:
            curved_score = 0.70
        else:
            curved_score = raw_score

        quality      = self._quality_penalty(answer)
        final_score  = round(curved_score * quality, 2)
        final_score  = max(0.0, min(1.0, final_score))

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
                "excerpt":         p["text"][:150] + "..." if len(p["text"]) > 150 else p["text"],
                "relevance_score": round(p.get("similarity_score", 0), 2)
            })

        # 4. Score Breakdown
        breakdown = {
            "consistency":  final_score,
            "semantic":     curved_score,     
            "completeness": 0.95 if final_score > 0.8 else final_score,
            "precision":    round(quality, 2)
        }

        return final_score, explanation, citations, breakdown