import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))


def format_chat_sample(question: str, answer: str) -> dict:
    """
    Formats a single Q&A pair into LLaMA-3 Instruct chat format.
    This is the exact format the model was originally trained on.
    """
    return {
        "text": (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a helpful, intelligent, and professional AI assistant. "
            f"Provide accurate, well-structured answers based on verified information.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question.strip()}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{answer.strip()}<|eot_id|>"
        )
    }


def format_hard_negative_sample(question: str, answer: str) -> dict:
    """
    Formats a hard negative sample.
    Wraps the bad answer with a humility prefix so the model learns
    to be cautious on similar questions in the future.
    """
    humility_prefix = (
        "I want to be transparent: my response on this topic may not have "
        "been fully accurate or helpful. Here is what I previously said, "
        "and I encourage you to verify this information: "
    )
    return {
        "text": (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a helpful, intelligent, and professional AI assistant. "
            f"When uncertain, acknowledge limitations clearly.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{question.strip()}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{humility_prefix}{answer.strip()}<|eot_id|>"
        )
    }


def load_jsonl(filepath: str) -> list:
    """Loads a JSONL file and returns list of dicts."""
    path = Path(filepath)
    if not path.exists():
        print(f"  [WARNING] File not found: {filepath} — skipping.")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  Loaded {len(records)} records from {filepath}")
    return records


def format_training_data(
    gold_path:   str = "data/gold_standard.jsonl",
    hard_path:   str = "data/hard_negatives.jsonl",
    output_path: str = "data/training_data_final.jsonl"
):
    """
    Merges gold standard + hard negatives into a single
    LLaMA-3 formatted training file ready for fine-tuning.
    """

    print("\n[Format Training Data] Starting...")
    print(f"  Gold standard  : {gold_path}")
    print(f"  Hard negatives : {hard_path}")
    print(f"  Output         : {output_path}")
    print("")

    # ── Load both datasets ─────────────────────────────────────────────────
    gold_records = load_jsonl(gold_path)
    hard_records = load_jsonl(hard_path)

    if not gold_records and not hard_records:
        print("[Format Training Data] ERROR: No data found in either file.")
        print("  Run extract_gold_standard.py and extract_hard_negatives.py first.")
        return 0

    # ── Format all samples ─────────────────────────────────────────────────
    formatted_samples = []

    # Gold standard → teach model correct, confident answers
    gold_count = 0
    for record in gold_records:
        question = record.get("question", "").strip()
        answer   = record.get("answer", "").strip()
        if question and answer:
            formatted_samples.append(format_chat_sample(question, answer))
            gold_count += 1

    # Hard negatives → teach model humility on bad answers
    hard_count = 0
    for record in hard_records:
        question = record.get("question", "").strip()
        answer   = record.get("answer", "").strip()
        if question and answer:
            formatted_samples.append(format_hard_negative_sample(question, answer))
            hard_count += 1

    print(f"\n[Format Training Data] Formatted samples:")
    print(f"  Gold standard  : {gold_count} samples")
    print(f"  Hard negatives : {hard_count} samples")
    print(f"  Total          : {len(formatted_samples)} samples")

    if not formatted_samples:
        print("[Format Training Data] ERROR: No valid samples after formatting.")
        return 0

    # ── Save final JSONL ───────────────────────────────────────────────────
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in formatted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[Format Training Data] Saved {len(formatted_samples)} samples to: {output_file}")

    # ── Preview first sample ───────────────────────────────────────────────
    print("\n[Format Training Data] Preview of first sample:")
    print("-" * 60)
    print(formatted_samples[0]["text"][:300] + "...")
    print("-" * 60)

    return len(formatted_samples)


if __name__ == "__main__":
    start = datetime.now()

    total = format_training_data(
        gold_path   = "data/gold_standard.jsonl",
        hard_path   = "data/hard_negatives.jsonl",
        output_path = "data/training_data_final.jsonl"
    )

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n[Format Training Data] Done! {total} samples in {elapsed:.2f}s")
    print(f"[Format Training Data] File ready for LLaMA fine-tuning!")
