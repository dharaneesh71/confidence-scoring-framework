import json
import sys
import math
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ==========================================
# BLEU SCORE
# ==========================================

def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _clip_count(candidate_ngrams: Counter, reference_ngrams: Counter) -> int:
    clipped = 0
    for ngram, count in candidate_ngrams.items():
        clipped += min(count, reference_ngrams.get(ngram, 0))
    return clipped


def compute_bleu(reference: str, hypothesis: str, max_n: int = 2) -> float:
    """BLEU-2 score between reference and hypothesis. Returns 0.0 to 1.0."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / len(hyp_tokens)
    )

    log_score = 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(ref_tokens, n)
        hyp_ngrams = _ngrams(hyp_tokens, n)
        total      = sum(hyp_ngrams.values())
        clipped    = _clip_count(hyp_ngrams, ref_ngrams)

        if total == 0 or clipped == 0:
            return 0.0

        log_score += math.log(clipped / total)

    return round(bp * math.exp(log_score / max_n), 4)


# ==========================================
# ROUGE SCORE
# ==========================================

def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L. Returns F1 scores 0.0 to 1.0."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    def f1(match, ref_len, hyp_len):
        precision = match / hyp_len if hyp_len else 0.0
        recall    = match / ref_len if ref_len  else 0.0
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    # ROUGE-1
    ref_1 = Counter(ref_tokens)
    hyp_1 = Counter(hyp_tokens)
    match_1 = sum((ref_1 & hyp_1).values())
    rouge1  = f1(match_1, len(ref_tokens), len(hyp_tokens))

    # ROUGE-2
    ref_2   = _ngrams(ref_tokens, 2)
    hyp_2   = _ngrams(hyp_tokens, 2)
    match_2 = sum((ref_2 & hyp_2).values())
    rouge2  = f1(match_2, sum(ref_2.values()), sum(hyp_2.values()))

    # ROUGE-L (LCS)
    m, n = len(ref_tokens), len(hyp_tokens)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs    = dp[m][n]
    rougeL = f1(lcs, len(ref_tokens), len(hyp_tokens))

    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}


# ==========================================
# MAIN EVALUATION
# ==========================================

def evaluate_models(
    training_data_path: str = "data/training_data_final.jsonl",
    output_path:        str = "data/evaluation_results.json"
):
    print("\n[Evaluate Models] Starting...")
    print(f"  Input  : {training_data_path}")
    print(f"  Output : {output_path}\n")

    training_file = Path(training_data_path)
    if not training_file.exists():
        print("[Evaluate Models] ERROR: training_data_final.jsonl not found.")
        print("  Run format_training_data.py first.")
        return None

    samples = []
    with open(training_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"  Loaded {len(samples)} samples.\n")

    from core.database import SessionLocal, ChatHistory, Feedback
    db = SessionLocal()
    try:
        gold_rows = (
            db.query(ChatHistory)
            .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
            .filter(Feedback.rating >= 4)
            .all()
        )
        hard_rows = (
            db.query(ChatHistory)
            .join(Feedback, Feedback.chat_history_id == ChatHistory.id)
            .filter(Feedback.rating <= 2)
            .all()
        )
    finally:
        db.close()

    # Build reference map: question → {answer, confidence, type}
    reference_map = {}
    for row in gold_rows:
        reference_map[row.question.strip()] = {
            "answer":     row.answer,
            "confidence": row.confidence_score,
            "type":       "gold_standard",
        }
    for row in hard_rows:
        reference_map[row.question.strip()] = {
            "answer":     row.answer,
            "confidence": row.confidence_score,
            "type":       "hard_negative",
        }

    results      = []
    total_bleu   = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    count        = 0

    for sample in samples:
        raw_text = sample.get("text", "")
        try:
            user_part      = raw_text.split("<|start_header_id|>user<|end_header_id|>\n\n")[1]
            question       = user_part.split("<|eot_id|>")[0].strip()
            assistant_part = raw_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1]
            hypothesis     = assistant_part.split("<|eot_id|>")[0].strip()
        except IndexError:
            continue

        ref_entry = reference_map.get(question)
        if not ref_entry:
            continue

        reference = ref_entry["answer"]
        bleu      = compute_bleu(reference, hypothesis)
        rouge     = compute_rouge(reference, hypothesis)

        total_bleu   += bleu
        total_rouge1 += rouge["rouge1"]
        total_rouge2 += rouge["rouge2"]
        total_rougeL += rouge["rougeL"]
        count        += 1

        results.append({
            "question":   question[:80] + "..." if len(question) > 80 else question,
            "type":       ref_entry["type"],
            "confidence": round(ref_entry["confidence"], 4),
            "bleu":       bleu,
            "rouge1":     rouge["rouge1"],
            "rouge2":     rouge["rouge2"],
            "rougeL":     rouge["rougeL"],
        })

        print(f"  Q : {question[:55]}...")
        print(f"  Type       : {ref_entry['type']}")
        print(f"  BLEU       : {bleu}")
        print(f"  ROUGE-1    : {rouge['rouge1']}")
        print(f"  ROUGE-2    : {rouge['rouge2']}")
        print(f"  ROUGE-L    : {rouge['rougeL']}")
        print(f"  Confidence : {round(ref_entry['confidence'], 4)}\n")

    if count == 0:
        print("[Evaluate Models] No matching samples found.")
        return None

    avg_bleu   = round(total_bleu   / count, 4)
    avg_rouge1 = round(total_rouge1 / count, 4)
    avg_rouge2 = round(total_rouge2 / count, 4)
    avg_rougeL = round(total_rougeL / count, 4)

    print("=" * 55)
    print(f"[Evaluate Models] AVERAGES over {count} samples:")
    print(f"  BLEU    : {avg_bleu}")
    print(f"  ROUGE-1 : {avg_rouge1}")
    print(f"  ROUGE-2 : {avg_rouge2}")
    print(f"  ROUGE-L : {avg_rougeL}")
    print("=" * 55)

    output = {
        "evaluated_at":  datetime.now().isoformat(),
        "total_samples": count,
        "averages": {
            "bleu":   avg_bleu,
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
        },
        "per_sample": results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Evaluate Models] Saved to: {output_file}")
    return output


if __name__ == "__main__":
    start  = datetime.now()
    evaluate_models(
        training_data_path="data/training_data_final.jsonl",
        output_path="data/evaluation_results.json"
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n[Evaluate Models] Done in {elapsed:.2f}s")
