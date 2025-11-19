"""
Enhanced Confidence Scoring Service with Multi-Dimensional Evaluation
Returns score breakdown for frontend display
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
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


class ScoringService:
    """
    Enhanced Confidence Scoring Engine that evaluates AI-generated answers
    against Ground Truth Dataset using four dimensions:
    1. Factual Consistency
    2. Semantic Alignment
    3. Completeness
    4. Precision (Hallucination Detection)
    """
    
    def __init__(self):
        """Initialize scoring service with embedding model"""
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Enhanced Scoring Service initialized")
        
        # Uncertainty phrases that indicate honest admission of not knowing
        self.uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot", "i am not sure",
            "no information", "unclear", "uncertain", "not enough information",
            "unable to determine", "cannot confirm", "insufficient data",
            "i do not have", "no data available", "cannot provide"
        ]
    
    def compute_confidence_score(
        self,
        answer: str,
        question: str,
        retrieved_passages: List[Dict]
    ) -> Tuple[float, str, List[Dict], Dict[str, float]]:
        """
        Compute comprehensive confidence score using four dimensions
        
        Args:
            answer: AI-generated answer
            question: Original user question
            retrieved_passages: List of relevant passages from knowledge base
            
        Returns:
            Tuple of (confidence_score, explanation, scored_citations, score_breakdown)
        """
        if not answer or not retrieved_passages:
            return 0.0, "Insufficient information to compute confidence", [], {}
        
        try:
            # Check if answer expresses honest uncertainty
            is_uncertain = self._check_uncertainty(answer)
            
            # 1. Factual Consistency (35% weight)
            consistency_score, consistency_details = self._compute_factual_consistency(
                answer, retrieved_passages
            )
            
            # 2. Semantic Alignment (30% weight)
            semantic_score, semantic_details = self._compute_semantic_alignment(
                answer, retrieved_passages
            )
            
            # 3. Completeness (25% weight)
            completeness_score, completeness_details = self._compute_completeness(
                answer, question, retrieved_passages
            )
            
            # 4. Precision - Hallucination Detection (10% weight)
            precision_score, precision_details = self._compute_precision(
                answer, retrieved_passages, is_uncertain
            )
            
            # Weighted combination of all scores
            weights = {
                'consistency': 0.35,
                'semantic': 0.30,
                'completeness': 0.25,
                'precision': 0.10
            }
            
            final_score = (
                weights['consistency'] * consistency_score +
                weights['semantic'] * semantic_score +
                weights['completeness'] * completeness_score +
                weights['precision'] * precision_score
            )
            
            # Apply edge case adjustments
            final_score = self._apply_edge_case_adjustments(
                final_score, answer, retrieved_passages, is_uncertain,
                consistency_score, semantic_score, completeness_score, precision_score
            )
            
            # Round to 2 decimal places
            final_score = round(min(max(final_score, 0.0), 1.0), 2)
            
            # Create score breakdown for frontend
            score_breakdown = {
                'consistency': round(consistency_score, 2),
                'semantic': round(semantic_score, 2),
                'completeness': round(completeness_score, 2),
                'precision': round(precision_score, 2)
            }
            
            # Generate detailed explanation
            explanation = self._generate_detailed_explanation(
                final_score,
                is_uncertain,
                {
                    'consistency': (consistency_score, consistency_details),
                    'semantic': (semantic_score, semantic_details),
                    'completeness': (completeness_score, completeness_details),
                    'precision': (precision_score, precision_details)
                }
            )
            
            # Create scored citations
            citations = self._create_citations(retrieved_passages)
            
            logger.info(f"Computed confidence score: {final_score} (Uncertain: {is_uncertain})")
            logger.info(f"Breakdown - Consistency: {score_breakdown['consistency']}, "
                       f"Semantic: {score_breakdown['semantic']}, "
                       f"Completeness: {score_breakdown['completeness']}, "
                       f"Precision: {score_breakdown['precision']}")
            
            return final_score, explanation, citations, score_breakdown
            
        except Exception as e:
            logger.error(f"Error computing confidence score: {e}")
            return 0.5, "Error occurred during scoring", [], {}
    
    def _check_uncertainty(self, answer: str) -> bool:
        """Check if answer expresses honest uncertainty"""
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in self.uncertainty_phrases)
    
    def _compute_factual_consistency(
        self, answer: str, passages: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Dimension 1: Factual Consistency
        Every claim must be explicitly supported by retrieved evidence
        """
        try:
            # Split answer into individual sentences (claims)
            answer_sentences = sent_tokenize(answer)
            
            if not answer_sentences:
                return 0.0, {"supported": 0, "total": 0}
            
            # Encode answer sentences
            answer_embeddings = self.embedding_model.encode(
                answer_sentences, convert_to_tensor=True
            )
            
            # Encode passages
            passage_texts = [p["text"] for p in passages]
            passage_embeddings = self.embedding_model.encode(
                passage_texts, convert_to_tensor=True
            )
            
            # For each claim, check if it's supported by any passage
            supported_count = 0
            claim_scores = []
            
            for sent_emb in answer_embeddings:
                similarities = util.cos_sim(sent_emb, passage_embeddings)[0]
                max_similarity = float(similarities.max())
                claim_scores.append(max_similarity)
                
                # Consider a claim supported if similarity > 0.6
                if max_similarity >= 0.60:
                    supported_count += 1
            
            # Calculate consistency score
            consistency_score = np.mean(claim_scores) if claim_scores else 0.0
            
            details = {
                "supported_claims": supported_count,
                "total_claims": len(answer_sentences),
                "support_ratio": supported_count / len(answer_sentences),
                "avg_similarity": round(consistency_score, 2)
            }
            
            return consistency_score, details
            
        except Exception as e:
            logger.error(f"Error computing factual consistency: {e}")
            return 0.5, {}
    
    def _compute_semantic_alignment(
        self, answer: str, passages: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Dimension 2: Semantic Alignment
        Meaning must match ground truth even if wording differs
        """
        try:
            # Encode full answer
            answer_embedding = self.embedding_model.encode(answer, convert_to_tensor=True)
            
            # Encode passages
            passage_texts = [p["text"] for p in passages]
            passage_embeddings = self.embedding_model.encode(
                passage_texts, convert_to_tensor=True
            )
            
            # Compute cosine similarities
            similarities = util.cos_sim(answer_embedding, passage_embeddings)[0]
            
            max_similarity = float(similarities.max())
            avg_top3_similarity = float(similarities.topk(min(3, len(similarities)))[0].mean())
            
            # Weighted combination: prioritize best match but consider consistency
            semantic_score = 0.6 * max_similarity + 0.4 * avg_top3_similarity
            
            details = {
                "max_similarity": round(max_similarity, 2),
                "avg_top3_similarity": round(avg_top3_similarity, 2),
                "alignment_level": "strong" if semantic_score > 0.75 else 
                                  "moderate" if semantic_score > 0.55 else "weak"
            }
            
            return semantic_score, details
            
        except Exception as e:
            logger.error(f"Error computing semantic alignment: {e}")
            return 0.5, {}
    
    def _compute_completeness(
        self, answer: str, question: str, passages: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Dimension 3: Completeness
        Answer should address all essential parts of the question
        """
        try:
            # Extract key concepts from question
            question_words = set(word_tokenize(question.lower()))
            question_keywords = {w for w in question_words if len(w) > 3}
            
            # Extract concepts from answer
            answer_words = set(word_tokenize(answer.lower()))
            answer_keywords = {w for w in answer_words if len(w) > 3}
            
            # Extract concepts from ground truth passages
            passage_text = " ".join([p["text"] for p in passages])
            passage_words = set(word_tokenize(passage_text.lower()))
            passage_keywords = {w for w in passage_words if len(w) > 3}
            
            # Check question keyword coverage in answer
            question_coverage = 0.0
            if question_keywords:
                covered_keywords = question_keywords & answer_keywords
                question_coverage = len(covered_keywords) / len(question_keywords)
            
            # Check if answer covers key concepts from ground truth
            essential_concepts = passage_keywords & question_keywords
            if essential_concepts:
                covered_essential = essential_concepts & answer_keywords
                essential_coverage = len(covered_essential) / len(essential_concepts)
            else:
                essential_coverage = 0.5
            
            # Check answer length appropriateness
            answer_sentences = sent_tokenize(answer)
            passage_sentences = sent_tokenize(passage_text)
            
            length_ratio = len(answer_sentences) / max(len(passage_sentences), 1)
            length_score = 1.0 if 0.3 <= length_ratio <= 1.2 else 0.7
            
            # Detect definition-style questions (require more complete answers)
            is_definition_question = any(
                keyword in question.lower() 
                for keyword in ["what is", "define", "properties", "characteristics", "features"]
            )
            
            # Combine scores
            if is_definition_question:
                # Penalize incompleteness more for definitions
                completeness_score = (
                    0.3 * question_coverage +
                    0.5 * essential_coverage +
                    0.2 * length_score
                )
            else:
                completeness_score = (
                    0.4 * question_coverage +
                    0.4 * essential_coverage +
                    0.2 * length_score
                )
            
            details = {
                "question_coverage": round(question_coverage, 2),
                "essential_coverage": round(essential_coverage, 2),
                "is_definition_question": is_definition_question,
                "completeness_level": "high" if completeness_score > 0.7 else
                                     "moderate" if completeness_score > 0.5 else "low"
            }
            
            return completeness_score, details
            
        except Exception as e:
            logger.error(f"Error computing completeness: {e}")
            return 0.5, {}
    
    def _compute_precision(
        self, answer: str, passages: List[Dict], is_uncertain: bool
    ) -> Tuple[float, Dict]:
        """
        Dimension 4: Precision (No Hallucination)
        Answer must not include fabricated or irrelevant details
        If uncertainty is expressed, give high precision
        """
        try:
            # If model admits uncertainty honestly, give maximum precision
            if is_uncertain:
                return 1.0, {
                    "hallucination_risk": "none",
                    "reason": "honest_uncertainty",
                    "precision_level": "high"
                }
            
            # Extract meaningful terms from answer (excluding stopwords)
            answer_words = set(word_tokenize(answer.lower()))
            answer_terms = {w for w in answer_words if len(w) > 3 and w.isalnum()}
            
            # Extract terms from passages
            passage_text = " ".join([p["text"] for p in passages])
            passage_words = set(word_tokenize(passage_text.lower()))
            passage_terms = {w for w in passage_words if len(w) > 3 and w.isalnum()}
            
            if not answer_terms:
                return 0.5, {"hallucination_risk": "unknown"}
            
            # Calculate term support ratio
            supported_terms = answer_terms & passage_terms
            unsupported_terms = answer_terms - passage_terms
            
            support_ratio = len(supported_terms) / len(answer_terms)
            
            # Check for numeric hallucinations
            answer_numbers = set([w for w in answer_words if w.replace('.', '').isdigit()])
            passage_numbers = set([w for w in passage_words if w.replace('.', '').isdigit()])
            
            numeric_hallucination = False
            if answer_numbers and not (answer_numbers & passage_numbers):
                numeric_hallucination = True
                support_ratio *= 0.7  # Penalty for numeric hallucination
            
            # Sentence-level hallucination check
            answer_sentences = sent_tokenize(answer)
            if len(answer_sentences) > 0:
                answer_sent_embeddings = self.embedding_model.encode(
                    answer_sentences, convert_to_tensor=True
                )
                passage_embeddings = self.embedding_model.encode(
                    [p["text"] for p in passages], convert_to_tensor=True
                )
                
                low_similarity_count = 0
                for sent_emb in answer_sent_embeddings:
                    similarities = util.cos_sim(sent_emb, passage_embeddings)[0]
                    if float(similarities.max()) < 0.4:
                        low_similarity_count += 1
                
                hallucination_ratio = low_similarity_count / len(answer_sentences)
                
                # Combine term-level and sentence-level precision
                precision_score = 0.6 * support_ratio + 0.4 * (1 - hallucination_ratio)
            else:
                precision_score = support_ratio
            
            details = {
                "supported_terms": len(supported_terms),
                "unsupported_terms": len(unsupported_terms),
                "support_ratio": round(support_ratio, 2),
                "numeric_hallucination": numeric_hallucination,
                "hallucination_risk": "low" if precision_score > 0.7 else
                                     "moderate" if precision_score > 0.5 else "high",
                "precision_level": "high" if precision_score > 0.7 else
                                  "moderate" if precision_score > 0.5 else "low"
            }
            
            return precision_score, details
            
        except Exception as e:
            logger.error(f"Error computing precision: {e}")
            return 0.5, {}
    
    def _apply_edge_case_adjustments(
        self, base_score: float, answer: str, passages: List[Dict],
        is_uncertain: bool, consistency: float, semantic: float,
        completeness: float, precision: float
    ) -> float:
        """
        Apply adjustments for edge cases:
        - Partially correct answers with missing essential details
        - Answers with extra incorrect claims
        - Honest uncertainty
        """
        adjusted_score = base_score
        
        # Case 1: Honest uncertainty - maintain reasonable score based on precision
        if is_uncertain:
            return adjusted_score
        
        # Case 2: Low completeness with high semantic similarity
        if semantic > 0.6 and completeness < 0.6:
            adjusted_score = min(adjusted_score, 0.65)
            logger.info("Applied partial correctness penalty")
        
        # Case 3: Low precision indicates hallucination
        if precision < 0.5:
            adjusted_score *= 0.7
            logger.info("Applied hallucination penalty")
        
        # Case 4: All dimensions low - likely wrong answer
        if all(score < 0.5 for score in [consistency, semantic, completeness, precision]):
            adjusted_score = min(adjusted_score, 0.40)
            logger.info("Applied low confidence cap")
        
        # Case 5: High semantic but low consistency
        if semantic > 0.7 and consistency < 0.5:
            adjusted_score = min(adjusted_score, 0.55)
            logger.info("Applied unsupported claims penalty")
        
        return adjusted_score
    
    def _generate_detailed_explanation(
        self, final_score: float, is_uncertain: bool, components: Dict
    ) -> str:
        """Generate comprehensive explanation of the confidence score"""
        
        level = self._get_confidence_level(final_score)
        
        explanation = f"Confidence Level: {level} ({final_score}). "
        
        if is_uncertain:
            explanation += "The model honestly expressed uncertainty. "
        
        # Build detailed component analysis
        component_details = []
        
        # Factual Consistency
        cons_score, cons_details = components['consistency']
        if cons_score >= 0.7:
            component_details.append(
                f"strong factual support ({cons_details.get('supported_claims', 0)}/"
                f"{cons_details.get('total_claims', 0)} claims verified)"
            )
        elif cons_score >= 0.5:
            component_details.append("moderate factual support with some unverified claims")
        else:
            component_details.append("weak factual support - many claims lack evidence")
        
        # Semantic Alignment
        sem_score, sem_details = components['semantic']
        alignment = sem_details.get('alignment_level', 'unknown')
        component_details.append(f"{alignment} semantic alignment")
        
        # Completeness
        comp_score, comp_details = components['completeness']
        if comp_details.get('is_definition_question') and comp_score < 0.6:
            component_details.append("incomplete answer - missing essential properties")
        elif comp_score >= 0.7:
            component_details.append("comprehensive coverage of key concepts")
        elif comp_score < 0.5:
            component_details.append("limited coverage - important details missing")
        
        # Precision
        prec_score, prec_details = components['precision']
        risk = prec_details.get('hallucination_risk', 'unknown')
        if risk == "none":
            component_details.append("no hallucination detected")
        elif risk == "high":
            component_details.append("possible unsupported or fabricated claims")
        elif prec_details.get('numeric_hallucination'):
            component_details.append("potential numeric inaccuracies")
        
        explanation += "Based on: " + "; ".join(component_details) + "."
        
        # Add recommendation if score is low
        if final_score < 0.5:
            explanation += " Recommend verifying answer with authoritative sources."
        
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
        """Format passages as citations with relevance scores"""
        citations = []
        for idx, passage in enumerate(passages):
            citation = {
                "id": idx + 1,
                "source": passage.get("source", "unknown"),
                "page": passage.get("page", None),
                "excerpt": passage["text"][:200] + "..." if len(passage["text"]) > 200 else passage["text"],
                "similarity_score": round(passage.get("similarity_score", 0.0), 2),
                "relevance_score": round(passage.get("similarity_score", 0.0), 2)  # Alias
            }
            citations.append(citation)
        
        return citations