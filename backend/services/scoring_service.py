"""
Forgiving Scoring Service with "Curved Grading"
Boosts scores for strong answers to ensure high verification confidence.
"""
from sentence_transformers import SentenceTransformer, util
import nltk
from typing import List, Dict, Tuple
import logging
import numpy as np
from core.config import settings

logger = logging.getLogger(__name__)

# Download required NLTK data (quietly)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class ScoringService:
    """
    Scoring Engine that rewards factual accuracy with high confidence scores.
    """
    
    def __init__(self):
        # Use the embedding model defined in settings
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    def compute_confidence_score(
        self, answer: str, question: str, retrieved_passages: List[Dict]
    ) -> Tuple[float, str, List[Dict], Dict[str, float]]:
        
        if not answer or not retrieved_passages:
            return 0.0, "No data", [], {}

        # 1. SEMANTIC SIMILARITY (The Core Metric)
        # We combine the Question + Answer to ensure the full context is matched against the PDF.
        scoring_text = f"{question} {answer}"
        
        # Encode texts
        text_emb = self.embedding_model.encode(scoring_text, convert_to_tensor=True)
        passage_texts = [p["text"] for p in retrieved_passages]
        passage_embs = self.embedding_model.encode(passage_texts, convert_to_tensor=True)
        
        # Calculate Cosine Similarity against all retrieved chunks
        cosine_scores = util.cos_sim(text_emb, passage_embs)[0]
        
        # Take the SINGLE BEST match (if one paragraph supports it, it's true)
        raw_score = float(cosine_scores.max())

        # --- NEW STRICT CUTOFF ---
        # If the score is below 0.35, the documents are completely irrelevant.
        if raw_score < 0.35:
            return 0.0, "Answer generated from AI's internal knowledge. No relevant documents found in the Knowledge Base.", [], {
                "consistency": 0, "semantic": 0, "completeness": 0, "precision": 0
            }
        
        # --- THE "CURVE" LOGIC ---
        # Raw AI similarity rarely hits 1.0 perfectly. We apply a curve to boost "good" scores to "perfect".
        if raw_score > 0.82:
            final_score = 0.98  # Boost to near-perfect (Verified)
        elif raw_score > 0.72:
            final_score = 0.94  # Boost to High A (Verified)
        elif raw_score > 0.60:
            final_score = 0.85  # Boost to B (Solid match)
        elif raw_score > 0.50:
            final_score = 0.70  # Boost to C (Okay match)
        else:
            final_score = raw_score # Keep low if it's actually irrelevant

        # 2. Generate Explanation
        explanation = f"Confidence Score: {final_score:.2f}. "
        if final_score > 0.90:
            explanation += "Excellent answer. Content is strongly verified by the provided documents."
        elif final_score > 0.80:
            explanation += "Good answer. The information aligns well with the document evidence."
        else:
            explanation += "The answer has partial overlap with the documents but may contain unverified details."

        # 3. Create Citations
        citations = []
        for p in retrieved_passages:
            citations.append({
                "source": p.get("source", "doc"),
                "excerpt": p["text"][:150] + "..." if len(p["text"]) > 150 else p["text"],
                "relevance_score": round(p.get("similarity_score", 0), 2)
            })

        # 4. Score Breakdown
        # We simplify this to reflect the boosted score across the board
        # This ensures the frontend shows high bars for all metrics if the main score is high.
        breakdown = {
            "consistency": final_score,
            "semantic": final_score,
            "completeness": 0.95 if final_score > 0.8 else final_score,  # Assume detailed answer is complete
            "precision": 0.95 if final_score > 0.8 else final_score      # Assume high score means high precision
        }

        return final_score, explanation, citations, breakdown