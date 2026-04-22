"""
Llama Language Model Service — Fully Optimised
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    pipeline,
)
from huggingface_hub import login

from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    """Return the best available device string: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pipeline_device_kwargs(device: str) -> dict:
    """
    Return the correct device kwargs for the HF pipeline.
    device_map='auto' and an explicit device= must NEVER coexist —
    that combination causes a hard crash.
    """
    if device == "cuda":
        return {"device_map": "auto"}
    if device == "mps":
        return {"device": "mps"}
    return {"device": -1}   # CPU


def _model_load_kwargs(device: str) -> dict:
    """Return torch_dtype + placement kwargs for from_pretrained."""
    if device == "cuda":
        return {
            "torch_dtype": torch.float16,
            "device_map":  "auto",
        }
    if device == "mps":
        # float16 is numerically unstable on Apple Silicon — use float32
        return {"torch_dtype": torch.float32}
    return {"torch_dtype": torch.float32}


# ---------------------------------------------------------------------------
# Internal dataset (used only during fine-tuning)
# ---------------------------------------------------------------------------

class _ChatDataset(torch.utils.data.Dataset):
    """
    Stores pre-tokenised sequences as plain Python lists (NOT padded tensors).
    DataCollatorForLanguageModeling applies dynamic per-batch padding at
    training time, which is far more memory-efficient than static padding.
    """

    def __init__(self, encodings: dict):
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        return {
            "input_ids":      ids,
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels":         ids.clone(),   # causal-LM: labels == input_ids
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LlamaService:
    """Manages a LLaMA-3 model for professional answer generation."""

    # LLaMA-3 special tokens — defined once as class constants
    _ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    _EOT              = "<|eot_id|>"

    # Default generation hyper-parameters — single source of truth.
    # Stored here so retrain() can re-stamp them after save_pretrained()
    # resets the model's generation_config.
    _GEN_CONFIG = dict(
        max_new_tokens     = 2048,
        temperature        = 0.7,
        do_sample          = True,
        max_length         = None,
        top_p              = 0.9,
        repetition_penalty = 1.1,
    )

    def __init__(self):
        self.model:      Optional[AutoModelForCausalLM] = None
        self.tokenizer:  Optional[AutoTokenizer]        = None
        self.pipeline                                   = None

        self.device         = _resolve_device()
        self.model_name     = settings.LLAMA_MODEL_NAME
        self.finetuned_path = Path("data/finetuned_model")

        logger.info(f"LlamaService — device: '{self.device}'")
        self._initialize()

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def _initialize(self) -> None:
        try:
            if settings.HUGGINGFACE_TOKEN:
                login(token=settings.HUGGINGFACE_TOKEN)
                logger.info("HuggingFace login successful")

            load_path = (
                str(self.finetuned_path)
                if self.finetuned_path.exists()
                else self.model_name
            )
            logger.info(f"Loading model from: {load_path}")

            # ── Tokenizer ────────────────────────────────────────────────
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path,
                token=settings.HUGGINGFACE_TOKEN,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # ── Model ────────────────────────────────────────────────────
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                token=settings.HUGGINGFACE_TOKEN,
                **_model_load_kwargs(self.device),
            )

            if self.device == "mps":
                self.model = self.model.to("mps")

            # ── Stamp GenerationConfig ONCE on the model ─────────────────
            # All generation params live here and nowhere else.
            # The pipeline call passes ZERO extra kwargs, so there is no
            # "generation_config + kwargs" merge → warning is gone entirely.
            self._stamp_generation_config()

            # Disable gradient tracking for inference (~30 % memory saving)
            self.model.eval()

            self._rebuild_pipeline()
            logger.info("LlamaService initialised successfully")

        except Exception:
            logger.exception("Failed to initialise LlamaService")
            self.model    = None
            self.pipeline = None

    def _stamp_generation_config(self) -> None:
        """Apply _GEN_CONFIG onto model.generation_config (single source of truth)."""
        self.model.generation_config = GenerationConfig(
            **self._GEN_CONFIG,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _rebuild_pipeline(self) -> None:
        """Single source of truth for (re)creating the HF text-generation pipeline."""
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            **_pipeline_device_kwargs(self.device),
            **self._GEN_CONFIG
        )

    # ------------------------------------------------------------------ #
    #  Prompt builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, question: str, context: Optional[str]) -> str:
        """Return a fully-formed LLaMA-3 chat prompt string."""
        if context:
            user_text    = f"Based on this context: {context}\n\nQuestion: {question}"
            system_block = ""
        else:
            user_text    = question
            system_block = (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful, intelligent, and professional AI assistant. "
                "Provide a comprehensive, detailed, and well-structured answer. "
                "Explain concepts clearly as if teaching a student. "
                f"Answer directly without mentioning multiple-choice framing.{self._EOT}\n"
            )

        return (
            f"{system_block}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text}{self._EOT}\n"
            f"{self._ASSISTANT_HEADER}"
        )

    def _format_chat(self, question: str, answer: str) -> str:
        """Encode a Q&A pair into LLaMA-3 chat format for fine-tuning."""
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}{self._EOT}\n"
            f"{self._ASSISTANT_HEADER}"
            f"{answer}{self._EOT}"
        )

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a detailed answer for `question`, optionally grounded in `context`.

        Generation parameters are baked into model.generation_config at init
        time via _stamp_generation_config(). The pipeline call passes NO extra
        kwargs, which fully eliminates the HuggingFace deprecation warning:
          'Passing generation_config together with generation-related arguments'
        """
        if not self.is_ready():
            logger.warning("generate_answer called but model is not loaded")
            return "Model not loaded."

        prompt = self._build_prompt(question, context)

        try:
            with torch.no_grad():
                response = self.pipeline(prompt,**self._GEN_CONFIG)

            generated_text: str = response[0]["generated_text"]

            # Extract only the assistant turn from the full generated string
            if self._ASSISTANT_HEADER in generated_text:
                answer = generated_text.split(self._ASSISTANT_HEADER)[-1]
            else:
                answer = generated_text

            answer = answer.replace(self._EOT, "").strip()
            logger.info(f"Generated answer — {len(answer)} chars")
            return answer

        except Exception:
            logger.exception("Error in generate_answer")
            return "Error generating answer."

    # ------------------------------------------------------------------ #
    #  Fine-tuning                                                         #
    # ------------------------------------------------------------------ #

    def retrain(
        self,
        gold_data: List[dict],
        hard_data: List[dict],
        status_callback: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        """
        Fine-tune the loaded model on gold-standard and hard-negative data.
        Calls status_callback(progress_pct, message) at each stage.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded — cannot retrain.")

        def _cb(progress: int, msg: str) -> None:
            logger.info(f"[Retrain {progress:3d}%] {msg}")
            if status_callback:
                status_callback(progress, msg)

        # ── 1. Format training samples ───────────────────────────────────
        _cb(10, "Formatting training samples into LLaMA-3 chat format…")

        texts: List[str] = []
        texts.extend(
            self._format_chat(item["question"], item["answer"])
            for item in gold_data
        )
        texts.extend(
            self._format_chat(
                item["question"],
                (
                    "I want to be transparent: my previous response on this topic "
                    f"may not have been fully accurate. {item['answer']}"
                ),
            )
            for item in hard_data
        )

        if not texts:
            raise ValueError("No training samples available after formatting.")

        total = len(texts)
        _cb(20, f"Formatted {total} samples — tokenising (dynamic padding)…")

        # ── 2. Tokenise — NO static padding ─────────────────────────────
        # Plain lists returned here; _ChatDataset converts per-item to tensor.
        # DataCollatorForLanguageModeling applies dynamic padding per batch.
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
        )
        dataset = _ChatDataset(tokenized)
        _cb(35, f"Tokenisation complete — {len(dataset)} samples ready")

        # ── 3. Training arguments ─────────────────────────────────────────
        output_dir = str(self.finetuned_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir                  = output_dir,
            num_train_epochs            = 1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            learning_rate               = 2e-5,
            weight_decay                = 0.01,
            save_strategy               = "no",
            logging_steps               = 5,
            fp16                        = (self.device == "cuda"),
            bf16                        = False,   
            no_cuda                     = (self.device == "cpu"),
            use_mps_device              = (self.device == "mps"),
            report_to                   = "none",
            dataloader_pin_memory       = (self.device == "cuda"),
        )
        _cb(45, "Training arguments configured — starting fine-tuning…")

        # ── 4. Train ──────────────────────────────────────────────────────
        self.model.train()   # switch to train mode for the duration

        trainer = Trainer(
            model         = self.model,
            args          = training_args,
            train_dataset = dataset,
            data_collator = DataCollatorForLanguageModeling(
                tokenizer          = self.tokenizer,
                mlm                = False,  # causal LM, not masked
                pad_to_multiple_of = 8,      # tensor-core alignment on CUDA
            ),
        )
        trainer.train()
        _cb(80, "Fine-tuning complete — saving weights…")

        # ── 5. Persist ───────────────────────────────────────────────────
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        _cb(90, "Weights saved — restoring inference mode…")

        # ── 6. Restore inference state ────────────────────────────────────
        # save_pretrained() resets model.generation_config to defaults,
        # so we must re-stamp our custom config before rebuilding pipeline.
        self._stamp_generation_config()
        self.model.eval()
        self._rebuild_pipeline()
        _cb(100, "Pipeline live with newly fine-tuned weights!")
        logger.info(f"Retrain complete — model saved to '{output_dir}'")

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        """Return True only when both model and pipeline are loaded."""
        return self.model is not None and self.pipeline is not None