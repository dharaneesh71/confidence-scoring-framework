"""
Llama 3.1 Language Model service
Modified for "ChatGPT-Style" comprehensive answers (Blind Verification Mode)
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Optional
import logging
from core.config import settings

logger = logging.getLogger(__name__)


class LlamaService:
    """Manages Llama model for detailed, professional answer generation"""
    
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
            
            # Use the lighter model for speed/stability
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=settings.HUGGINGFACE_TOKEN
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=settings.HUGGINGFACE_TOKEN,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}")
            self.model = None
    
    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a detailed, helpful answer based on internal knowledge (Blind Mode).
        """
        try:
            if self.model is None:
                return "Model not loaded."
            
            # --- CHATGPT-STYLE PROMPT ---
            # We explicitly instruct the model to be detailed, educational, and professional.
            if context is None:
                prompt = (
                    f"<|start_header_id|>system<|end_header_id|>\n\n"
                    f"You are a helpful, intelligent, and professional AI assistant. "
                    f"Provide a comprehensive, detailed, and well-structured answer to the user's question. "
                    f"Explain the concepts clearly, as if you are teaching a student. "
                    f"Do not mention 'Option A' or 'multiple choice'. Just answer the question directly.<|eot_id|>\n"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{question}<|eot_id|>\n"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            else:
                # Fallback RAG prompt (only if you manually re-enable context later)
                prompt = (
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"Based on this context: {context}\n\n"
                    f"Question: {question}<|eot_id|>\n"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=512,  # Increased length for detailed answers
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Extract answer
            generated_text = response[0]['generated_text']
            
            # Clean up the response
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                answer = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
            elif "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1]
            else:
                answer = generated_text
            
            # Remove any trailing tokens
            answer = answer.replace("<|eot_id|>", "").strip()
            
            logger.info(f"Generated detailed answer of length {len(answer)}")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer."
    
    def is_ready(self) -> bool:
        return True