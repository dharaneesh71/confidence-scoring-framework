"""
Llama Language Model service
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from typing import Optional, Callable, List
import logging
import os
from pathlib import Path
from core.config import settings
from huggingface_hub import login

logger = logging.getLogger(__name__)

# ── Force offline mode to stop 503 HuggingFace ping errors ──
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"]       = "0"

class LlamaService:
    """Manages Llama model for detailed, professional answer generation"""

    def __init__(self):
        self.model          = None
        self.tokenizer      = None
        self.pipeline       = None
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_name     = settings.LLAMA_MODEL_NAME
        self.finetuned_path = Path("data/finetuned_model")
        self._initialize()

    def _initialize(self):
        try:
            # Only login if NOT in offline mode
            if os.environ.get("TRANSFORMERS_OFFLINE") != "1" and settings.HUGGINGFACE_TOKEN:
                login(token=settings.HUGGINGFACE_TOKEN)
                logger.info("HuggingFace login successful")

            # Use fine-tuned model if it exists, else base model
            load_path = str(self.finetuned_path) if self.finetuned_path.exists() else self.model_name
            logger.info(f"Loading model from: {load_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                token=settings.HUGGINGFACE_TOKEN,
                local_files_only=False,
                fix_mistral_regex=True
            )

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                token=settings.HUGGINGFACE_TOKEN,
                dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=False
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            logger.info("Llama model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}", exc_info=True)
            self.model    = None
            self.pipeline = None

    def _format_chat(self, question: str, answer: str) -> str:
        """Format a Q&A pair into LLaMA-3 chat format for fine-tuning."""
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{answer}<|eot_id|>"
        )

    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        try:
            if self.model is None:
                return "Model not loaded."

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
                prompt = (
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"Based on this context: {context}\n\n"
                    f"Question: {question}<|eot_id|>\n"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )

            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

            generated_text = response[0]["generated_text"]

            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                answer = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
            elif "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1]
            else:
                answer = generated_text

            answer = answer.replace("<|eot_id|>", "").strip()
            logger.info(f"Generated answer of length {len(answer)}")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer."

    def retrain(
        self,
        gold_data: List[dict],
        hard_data: List[dict],
        status_callback: Optional[Callable[[int, str], None]] = None
    ):
        """
        Sprint 5 Task 4+7: Fine-tunes the loaded model on gold standard
        and hard negative data using LoRA-style lightweight training.
        Calls status_callback(progress_pct, message) throughout.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded — cannot retrain.")

        def _cb(progress: int, msg: str):
            logger.info(f"[Retrain {progress}%] {msg}")
            if status_callback:
                status_callback(progress, msg)

        try:
            # ── Step 1: Format all training samples ──────────────────────
            _cb(15, "Formatting training samples into LLaMA-3 chat format...")
            texts = []

            for item in gold_data:
                texts.append(self._format_chat(item["question"], item["answer"]))

            # Hard negatives: teach model what NOT to be confident about
            for item in hard_data:
                corrected = (
                    f"I want to be transparent: my previous response on this topic "
                    f"may not have been fully accurate. {item['answer']}"
                )
                texts.append(self._format_chat(item["question"], corrected))

            if not texts:
                raise ValueError("No training samples available after formatting.")

            _cb(25, f"Formatted {len(texts)} training samples. Tokenizing...")

            # ── Step 2: Tokenize ─────────────────────────────────────────
            from torch.utils.data import Dataset

            class ChatDataset(Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings

                def __len__(self):
                    return len(self.encodings["input_ids"])

                def __getitem__(self, idx):
                    return {
                        "input_ids":      self.encodings["input_ids"][idx],
                        "attention_mask": self.encodings["attention_mask"][idx],
                        "labels":         self.encodings["input_ids"][idx].clone(),
                    }

            tokenized = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            dataset = ChatDataset(tokenized)
            _cb(40, "Tokenization complete. Configuring training...")

            # ── Step 3: Training arguments ───────────────────────────────
            output_dir = str(self.finetuned_path)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=2e-5,
                weight_decay=0.01,
                save_strategy="no",
                logging_steps=5,
                fp16=self.device == "cuda",
                no_cuda=self.device == "cpu",
                report_to="none",
            )
            _cb(50, "Starting fine-tuning pass...")

            # ── Step 4: Train ────────────────────────────────────────────
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                ),
            )
            trainer.train()
            _cb(80, "Fine-tuning complete. Saving new model weights...")

            # ── Step 5: Save fine-tuned model ────────────────────────────
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            _cb(90, "Model saved. Reloading pipeline with new weights...")

            # ── Step 6: Reload pipeline with updated model ───────────────
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            _cb(100, "Pipeline updated — new fine-tuned model is now live!")
            logger.info(f"Retrain complete. Model saved to {output_dir}")

        except Exception as e:
            logger.error(f"Retrain failed: {e}", exc_info=True)
            raise

    def is_ready(self) -> bool:
        return self.model is not None and self.pipeline is not None