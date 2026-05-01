"""
RetrainingService — orchestrates the full feedback-driven pipeline.

Pipeline steps (with progress percentages):
  Step 1  (20%) : Write gold-standard samples to JSONL
  Step 2  (40%) : Write hard-negative samples to JSONL
  Step 3  (60%) : Format both into LLaMA-3 chat format (training_data_final.jsonl)
  Step 4  (80%) : Evaluate with BLEU / ROUGE against DB reference answers
  Step 5  (90%) : Generate old-vs-new model comparison report
  Step 6 (100%) : Return metrics for model-version record

All file paths are derived from this file's location so the service works
regardless of the process working directory.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Absolute data directory ────────────────────────────────────────────────────
# retraining_service.py  →  backend/services/
# .parent                →  backend/services/
# .parent.parent         →  backend/
_SERVICES_DIR = Path(__file__).resolve().parent
_BACKEND_DIR  = _SERVICES_DIR.parent
DATA_DIR      = _BACKEND_DIR / "data"


class RetrainingService:
    """
    Orchestrates the full retraining pipeline:
    extract → format → evaluate → report → metrics.
    """

    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(
        self,
        gold_data: List[Dict],
        hard_data: List[Dict],
        status_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict:
        """
        Run the full pipeline and return a metrics dict suitable for
        LlamaService.save_model_version().

        Parameters
        ----------
        gold_data : list of dicts with keys question, answer, confidence_score
        hard_data : list of dicts with keys question, answer, confidence_score
        status_callback : callable(progress_pct: int, message: str) or None

        Returns
        -------
        dict with keys  accuracy, f1, loss
        """
        cb = status_callback if status_callback else lambda p, m: None

        # ── Step 1 : Persist gold-standard samples ─────────────────────────
        cb(10, "Step 1/6 — Saving gold-standard samples to JSONL…")
        gold_path = DATA_DIR / "gold_standard.jsonl"
        self._write_jsonl(gold_data, gold_path, record_type="gold_standard")
        cb(20, f"Step 1/6 — Gold standard: {len(gold_data)} samples saved.")
        logger.info("[Retrain] Step 1 complete — %d gold samples", len(gold_data))

        # ── Step 2 : Persist hard-negative samples ─────────────────────────
        cb(30, "Step 2/6 — Saving hard-negative samples to JSONL…")
        hard_path = DATA_DIR / "hard_negatives.jsonl"
        self._write_jsonl(hard_data, hard_path, record_type="hard_negative")
        cb(40, f"Step 2/6 — Hard negatives: {len(hard_data)} samples saved.")
        logger.info("[Retrain] Step 2 complete — %d hard samples", len(hard_data))

        # ── Step 3 : Format into LLaMA-3 chat format ───────────────────────
        cb(45, "Step 3/6 — Formatting training data into LLaMA-3 chat format…")
        train_path = DATA_DIR / "training_data_final.jsonl"
        n_formatted = self._format_training_data(gold_path, hard_path, train_path)
        cb(60, f"Step 3/6 — Formatted {n_formatted} training samples.")
        logger.info("[Retrain] Step 3 complete — %d formatted samples", n_formatted)

        # ── Step 4 : BLEU / ROUGE evaluation ──────────────────────────────
        cb(65, "Step 4/6 — Running BLEU / ROUGE evaluation…")
        eval_path = DATA_DIR / "evaluation_results.json"
        eval_results = self._evaluate_models(train_path, eval_path)
        n_eval = eval_results.get("total_samples", 0) if eval_results else 0
        cb(80, f"Step 4/6 — Evaluation complete: {n_eval} samples scored.")
        logger.info("[Retrain] Step 4 complete — %d evaluated", n_eval)

        # ── Step 5 : Comparison report ─────────────────────────────────────
        cb(85, "Step 5/6 — Generating model comparison report…")
        report_path = DATA_DIR / "model_comparison_report.json"
        self._generate_report(eval_path, report_path)
        cb(90, "Step 5/6 — Comparison report saved.")
        logger.info("[Retrain] Step 5 complete — report at %s", report_path)

        # ── Step 6 : Compute and return metrics ────────────────────────────
        cb(95, "Step 6/6 — Computing final metrics…")
        metrics = self._extract_metrics(eval_results)
        cb(100, (
            f"Step 6/6 — Pipeline complete. "
            f"BLEU={metrics['accuracy']:.4f}  "
            f"ROUGE-1={metrics['f1']:.4f}  "
            f"loss={metrics['loss']:.4f}"
        ))
        logger.info("[Retrain] Step 6 complete — metrics: %s", metrics)

        return metrics

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _write_jsonl(
        self,
        records: List[Dict],
        path: Path,
        record_type: str,
    ) -> None:
        """Write a list of Q&A dicts to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                entry = {
                    "type":             record_type,
                    "question":         rec.get("question", ""),
                    "answer":           rec.get("answer", ""),
                    "confidence_score": round(float(rec.get("confidence_score") or 0), 4),
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _format_training_data(
        self,
        gold_path: Path,
        hard_path: Path,
        output_path: Path,
    ) -> int:
        """
        Merge gold + hard JSONL files and format into LLaMA-3 Instruct chat
        format, writing to output_path.  Returns the number of samples written.
        """
        from scripts.format_training_data import (
            format_chat_sample,
            format_hard_negative_sample,
            load_jsonl,
        )

        gold_records = load_jsonl(str(gold_path))
        hard_records = load_jsonl(str(hard_path))

        formatted: List[Dict] = []

        for rec in gold_records:
            q = rec.get("question", "").strip()
            a = rec.get("answer", "").strip()
            if q and a:
                formatted.append(format_chat_sample(q, a))

        for rec in hard_records:
            q = rec.get("question", "").strip()
            a = rec.get("answer", "").strip()
            if q and a:
                formatted.append(format_hard_negative_sample(q, a))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for sample in formatted:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

        return len(formatted)

    def _evaluate_models(
        self,
        training_data_path: Path,
        output_path: Path,
    ) -> Optional[Dict]:
        """
        Run BLEU / ROUGE evaluation against DB reference answers.
        Returns the evaluation results dict, or None if no samples matched.
        """
        from scripts.evaluate_models import evaluate_models as _eval
        return _eval(
            training_data_path=str(training_data_path),
            output_path=str(output_path),
        )

    def _generate_report(
        self,
        evaluation_path: Path,
        output_path: Path,
    ) -> Optional[Dict]:
        """Generate old-vs-new comparison report from evaluation results."""
        from scripts.generate_reports import generate_report as _report
        return _report(
            evaluation_path=str(evaluation_path),
            output_path=str(output_path),
        )

    def _extract_metrics(self, eval_results: Optional[Dict]) -> Dict:
        """
        Map evaluation averages → model version metrics.

          accuracy  ← BLEU  (word-overlap precision)
          f1        ← ROUGE-1 F1
          loss      ← 1 − ROUGE-L  (inverse of LCS F1, lower is better)
        """
        if not eval_results:
            logger.warning("[Retrain] No evaluation results — returning zero metrics")
            return {"accuracy": 0.0, "f1": 0.0, "loss": 1.0}

        averages = eval_results.get("averages", {})
        bleu   = float(averages.get("bleu",   0.0))
        rouge1 = float(averages.get("rouge1", 0.0))
        rougeL = float(averages.get("rougeL", 0.0))

        return {
            "accuracy": round(bleu,          4),
            "f1":       round(rouge1,        4),
            "loss":     round(1.0 - rougeL,  4),
        }
