"""
Sprint 5 - Task 3
Script to extract "Hard Negatives" dataset (High Confidence + Low Rating)
Run: python -m scripts.extract_hard_negatives
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session
from core.database import SessionLocal, ChatHistory, Feedback


def extract_hard_negatives(
    min_confidence: float = 0.0, 
    max_rating: int = 2,
    output_path: str = "data/hard_negatives.jsonl"
):
    """
    Queries DB for entries where:
      - feedback rating <= max_rating   (1 or 2 stars = bad response)

    Hard negatives = any answer users rated poorly, regardless of confidence.
    These teach the model what NOT to do.
    """

    db: Session = SessionLocal()

    try:
        print(f"[Hard Negatives] Querying DB...")
        print(f"  Max rating : {max_rating}")

        # ── Only filter by rating, NOT confidence ──────────────────────────
        rows = (
            db.query(ChatHistory)
            .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
            .filter(Feedback.rating <= max_rating)
            .order_by(ChatHistory.timestamp.desc())
            .all()
        )

        if not rows:
            print("[Hard Negatives] No matching records found.")
            print("  Tip: Make sure users have submitted 1-2 star ratings.")
            return 0

        print(f"[Hard Negatives] Found {len(rows)} hard negative entries.")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for row in rows:
                record = {
                    "type":             "hard_negative",
                    "history_id":       row.id,
                    "question":         row.question,
                    "answer":           row.answer,
                    "confidence_score": round(row.confidence_score, 4),
                    "timestamp":        row.timestamp.isoformat() if row.timestamp else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"[Hard Negatives] Saved {count} records to: {output_file}")
        return count

    except Exception as e:
        print(f"[Hard Negatives] ERROR: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    start = datetime.now()
    total = extract_hard_negatives(
        min_confidence=0.0,
        max_rating=2,
        output_path="data/hard_negatives.jsonl"
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n[Hard Negatives] Done! {total} records in {elapsed:.2f}s")
    print(f"[Hard Negatives] Output: data/hard_negatives.jsonl")
