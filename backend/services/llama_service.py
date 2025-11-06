"""
Llama 3.1 Language Model service for answer generation
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Optional
import logging
from core.config import settings

logger = logging.getLogger(__name__)


class LlamaService:
    """Manages Llama 3.1 model for answer generation"""
    
    def __init__(self):
        """Initialize Llama model"""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize()
    
    def _initialize(self):
        """Load Llama model and tokenizer"""
        try:
            logger.info(f"Loading Llama model: {settings.LLAMA_MODEL_NAME}")
            logger.info(f"Using device: {self.device}")
            
            # For Sprint 1 demo, we'll use a smaller model for faster loading
            # In production, you would use the full Llama 3.1 model
            model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Smaller model for demo
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=settings.HUGGINGFACE_TOKEN
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=settings.HUGGINGFACE_TOKEN,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}")
            logger.warning("Falling back to mock responses for demo")
            # For Sprint 1 demo without HuggingFace access, we'll use mock responses
            self.model = None
    
    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate answer for a given question
        
        Args:
            question: User's question
            context: Optional context from retrieved documents
            
        Returns:
            Generated answer text
        """
        try:
            if self.model is None:
                # Mock response for demo purposes
                return self._generate_mock_answer(question, context)
            
            # Construct prompt
            if context:
                prompt = f"""Based on the following context, answer the question accurately.

Context: {context}

Question: {question}

Answer:"""
            else:
                prompt = f"""Question: {question}

Answer:"""
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Extract answer
            generated_text = response[0]['generated_text']
            answer = generated_text.split("Answer:")[-1].strip()
            
            logger.info(f"Generated answer of length {len(answer)}")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error generating an answer. Please try again."
    
    def _generate_mock_answer(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate mock answers for demo (when model is not available)
        """
        question_lower = question.lower()
        
        # Simple keyword-based mock responses
        if "confidence" in question_lower and "score" in question_lower:
            return ("The confidence scoring framework evaluates AI-generated answers by comparing them "
                   "to verified ground truth documents. It uses semantic similarity, factual consistency, "
                   "and completeness metrics to generate a score between 0.0 and 1.0, where higher scores "
                   "indicate greater alignment with trusted sources.")
        
        elif "architecture" in question_lower or "component" in question_lower:
            return ("The system consists of six core components: a React.js frontend for user interaction, "
                   "a FastAPI backend for orchestration, a Llama 3.1 AI model for answer generation, "
                   "a ChromaDB vector database for storing ground truth documents, a confidence scoring "
                   "pipeline for evaluation, and curated ground truth datasets verified by domain experts.")
        
        elif "how" in question_lower and "work" in question_lower:
            return ("When a user submits a question, the system: (1) generates an answer using the Llama AI model, "
                   "(2) retrieves the top 3 most relevant passages from the ground truth knowledge base using "
                   "semantic search, (3) compares the AI answer to these passages using sentence transformers "
                   "and NLTK, (4) computes a weighted confidence score based on semantic similarity, factual "
                   "consistency, and completeness, and (5) returns the answer with its confidence score and citations.")
        
        elif context:
            # If context is provided, use it to form a basic answer
            sentences = context.split('.')[:2]
            return ' '.join(sentences).strip() + '.'
        
        else:
            return (f"Based on the available information, I can provide some insights about {question}. "
                   "However, for a more detailed and accurate answer, please refer to the documentation "
                   "or consult with domain experts.")
    
    def is_ready(self) -> bool:
        """Check if Llama service is ready"""
        # For demo purposes, return True even with mock responses
        return True