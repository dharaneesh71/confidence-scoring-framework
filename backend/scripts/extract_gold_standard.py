import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session
from core.database import SessionLocal, ChatHistory, Feedback


def extract_gold_standard(
    min_confidence: float = 0.8,
    min_rating: int = 4,
    output_path: str = "data/gold_standard.jsonl"
):
    """
    Queries DB for entries where:
      - confidence_score >= min_confidence  (High Confidence)
      - feedback rating  >= min_rating      (High Rating = 4 or 5 stars)

    Saves results as JSONL file for LLaMA fine-tuning.
    """

    db: Session = SessionLocal()

    try:
        print(f"[Gold Standard] Querying DB...")
        print(f"  Min confidence : {min_confidence}")
        print(f"  Min rating     : {min_rating}")

        # ── Join ChatHistory with Feedback ─────────────────────────────────
        rows = (
            db.query(ChatHistory)
            .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
            .filter(ChatHistory.confidence_score >= min_confidence)
            .filter(Feedback.rating >= min_rating)
            .order_by(ChatHistory.timestamp.desc())
            .all()
        )

        if not rows:
            print("[Gold Standard] No matching records found.")
            print("  Tip: Make sure users have submitted feedback with 4-5 star ratings.")
            return 0

        print(f"[Gold Standard] Found {len(rows)} gold standard entries.")

        # ── Create output directory if needed ─────────────────────────────
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # ── Write JSONL ────────────────────────────────────────────────────
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for row in rows:
                record = {
                    "type":             "gold_standard",
                    "history_id":       row.id,
                    "question":         row.question,
                    "answer":           row.answer,
                    "confidence_score": round(row.confidence_score, 4),
                    "timestamp":        row.timestamp.isoformat() if row.timestamp else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"[Gold Standard] Saved {count} records to: {output_file}")
        return count

    except Exception as e:
        print(f"[Gold Standard] ERROR: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    total = extract_gold_standard(
        min_confidence=0.8,
        min_rating=4,
        output_path="data/gold_standard.jsonl"
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n[Gold Standard] Done! {total} records in {elapsed:.2f}s")
    print(f"[Gold Standard] Output: data/gold_standard.jsonl")