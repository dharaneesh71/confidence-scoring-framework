"""
Confidence scoring service for evaluating AI-generated answers
"""
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List, Dict, Tuple
import logging
import numpy as np
from core.config import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


class ScoringService:
    """Computes confidence scores for AI-generated answers"""
    
    def __init__(self):
        """Initialize scoring service with embedding model"""
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Scoring service initialized")
    
    def compute_confidence_score(
        self,
        answer: str,
        retrieved_passages: List[Dict]
    ) -> Tuple[float, str, List[Dict]]:
        """
        Compute comprehensive confidence score
        
        Args:
            answer: AI-generated answer
            retrieved_passages: List of relevant passages from knowledge base
            
        Returns:
            Tuple of (confidence_score, explanation, scored_citations)
        """
        if not answer or not retrieved_passages:
            return 0.0, "Insufficient information to compute confidence", []
        
        try:
            # 1. Semantic Similarity Score
            semantic_score = self._compute_semantic_similarity(answer, retrieved_passages)
            
            # 2. Coverage Score (how well passages cover the answer)
            coverage_score = self._compute_coverage_score(answer, retrieved_passages)
            
            # 3. Precision Score (detect potential hallucinations)
            precision_score = self._compute_precision_score(answer, retrieved_passages)
            
            # 4. Consistency Score (check factual alignment)
            consistency_score = self._compute_consistency_score(answer, retrieved_passages)
            
            # Weighted average of all scores
            weights = {
                'semantic': 0.35,
                'coverage': 0.25,
                'precision': 0.25,
                'consistency': 0.15
            }
            
            final_score = (
                weights['semantic'] * semantic_score +
                weights['coverage'] * coverage_score +
                weights['precision'] * precision_score +
                weights['consistency'] * consistency_score
            )
            
            # Round to 2 decimal places
            final_score = round(final_score, 2)
            
            # Generate explanation
            explanation = self._generate_explanation(
                final_score, semantic_score, coverage_score, 
                precision_score, consistency_score
            )
            
            # Create scored citations
            citations = self._create_citations(retrieved_passages)
            
            logger.info(f"Computed confidence score: {final_score}")
            return final_score, explanation, citations
            
        except Exception as e:
            logger.error(f"Error computing confidence score: {e}")
            return 0.5, "Error occurred during scoring", []
    
    def _compute_semantic_similarity(self, answer: str, passages: List[Dict]) -> float:
        """Compute semantic similarity between answer and passages"""
        try:
            # Encode answer
            answer_embedding = self.embedding_model.encode(answer, convert_to_tensor=True)
            
            # Encode passages
            passage_texts = [p["text"] for p in passages]
            passage_embeddings = self.embedding_model.encode(passage_texts, convert_to_tensor=True)
            
            # Compute cosine similarities
            similarities = util.cos_sim(answer_embedding, passage_embeddings)[0]
            
            # Take the maximum similarity (best matching passage)
            max_similarity = float(similarities.max())
            
            # Also consider average of top passages
            avg_similarity = float(similarities.mean())
            
            # Weighted combination
            score = 0.7 * max_similarity + 0.3 * avg_similarity
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.5
    
    def _compute_coverage_score(self, answer: str, passages: List[Dict]) -> float:
        """Check if key concepts in answer are covered by passages"""
        try:
            # Tokenize answer into sentences
            answer_sentences = sent_tokenize(answer)
            
            if not answer_sentences:
                return 0.0
            
            # Combine all passages
            combined_passages = " ".join([p["text"] for p in passages])
            
            # Check each answer sentence against passages
            sentence_scores = []
            for sentence in answer_sentences:
                # Simple keyword overlap check
                answer_words = set(word_tokenize(sentence.lower()))
                passage_words = set(word_tokenize(combined_passages.lower()))
                
                # Remove common stopwords
                answer_words = {w for w in answer_words if len(w) > 3}
                passage_words = {w for w in passage_words if len(w) > 3}
                
                if answer_words:
                    overlap = len(answer_words & passage_words) / len(answer_words)
                    sentence_scores.append(overlap)
            
            if sentence_scores:
                return np.mean(sentence_scores)
            return 0.5
            
        except Exception as e:
            logger.error(f"Error computing coverage score: {e}")
            return 0.5
    
    def _compute_precision_score(self, answer: str, passages: List[Dict]) -> float:
        """Detect potential hallucinations or unsupported claims"""
        try:
            # Check for uncertainty indicators (model admitting it doesn't know)
            uncertainty_phrases = [
                "i don't know", "i'm not sure", "i cannot", 
                "no information", "unclear", "uncertain"
            ]
            
            answer_lower = answer.lower()
            if any(phrase in answer_lower for phrase in uncertainty_phrases):
                # Model is honest about not knowing - high precision
                return 1.0
            
            # Extract key entities/claims from answer
            answer_words = set(word_tokenize(answer.lower()))
            answer_words = {w for w in answer_words if len(w) > 3}
            
            # Check if answer contains terms not in passages (potential hallucination)
            combined_passages = " ".join([p["text"] for p in passages]).lower()
            passage_words = set(word_tokenize(combined_passages))
            
            if not answer_words:
                return 0.5
            
            # Calculate what percentage of answer terms are found in passages
            supported_words = answer_words & passage_words
            precision = len(supported_words) / len(answer_words)
            
            return min(max(precision, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error computing precision score: {e}")
            return 0.5
    
    def _compute_consistency_score(self, answer: str, passages: List[Dict]) -> float:
        """Check for factual consistency"""
        try:
            # Use sentence-level similarity for consistency check
            answer_sentences = sent_tokenize(answer)
            
            if not answer_sentences:
                return 0.5
            
            # Encode answer sentences
            answer_embeddings = self.embedding_model.encode(answer_sentences, convert_to_tensor=True)
            
            # Encode passages
            passage_texts = [p["text"] for p in passages]
            passage_embeddings = self.embedding_model.encode(passage_texts, convert_to_tensor=True)
            
            # For each answer sentence, find best matching passage
            sentence_consistencies = []
            for sent_emb in answer_embeddings:
                similarities = util.cos_sim(sent_emb, passage_embeddings)[0]
                max_sim = float(similarities.max())
                sentence_consistencies.append(max_sim)
            
            # Average consistency across sentences
            return np.mean(sentence_consistencies)
            
        except Exception as e:
            logger.error(f"Error computing consistency score: {e}")
            return 0.5
    
    def _generate_explanation(
        self, final_score: float, semantic: float, 
        coverage: float, precision: float, consistency: float
    ) -> str:
        """Generate human-readable explanation of the score"""
        
        level = self._get_confidence_level(final_score)
        
        explanation = f"Confidence level: {level}. "
        
        # Add component breakdowns
        components = []
        if semantic > 0.8:
            components.append("strong semantic alignment with source material")
        elif semantic > 0.6:
            components.append("moderate semantic alignment")
        else:
            components.append("weak semantic alignment")
        
        if coverage > 0.7:
            components.append("good coverage of key concepts")
        elif coverage < 0.5:
            components.append("limited concept coverage")
        
        if precision < 0.6:
            components.append("possible unsupported claims")
        
        explanation += "Based on: " + ", ".join(components) + "."
        
        return explanation
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert numerical score to confidence label"""
        if score >= settings.HIGH_CONFIDENCE_THRESHOLD:
            return "High"
        elif score >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
            return "Medium"
        else:
            return "Low"
    
    def _create_citations(self, passages: List[Dict]) -> List[Dict]:
        """Format passages as citations"""
        citations = []
        for passage in passages:
            citation = {
                "source": passage.get("source", "unknown"),
                "page": passage.get("page", None),
                "excerpt": passage["text"][:200] + "..." if len(passage["text"]) > 200 else passage["text"],
                "similarity_score": round(passage.get("similarity_score", 0.0), 2)
            }
            citations.append(citation)
        
        return citations