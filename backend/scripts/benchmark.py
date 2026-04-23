"""
Task 21 — Performance benchmark: measures average response time per query.
Usage: python scripts/benchmark.py
"""
import time
import requests
import statistics

BASE_URL   = "http://localhost:8000"
TEST_EMAIL = "benchmark@test.com"
TEST_PASS  = "benchpass"

QUESTIONS = [
    "What is NLP?",
    "Explain machine learning in simple terms.",
    "What is a neural network?",
    "How does confidence scoring work?",
    "What is transfer learning?",
]


def get_token() -> str:
    requests.post(f"{BASE_URL}/api/auth/register",
                  json={"email": TEST_EMAIL, "password": TEST_PASS})
    res = requests.post(f"{BASE_URL}/api/auth/login",
                        data={"username": TEST_EMAIL, "password": TEST_PASS})
    return res.json().get("access_token", "")


def run_benchmark():
    print("Starting performance benchmark...\n")
    token   = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    results = []

    for q in QUESTIONS:
        start = time.time()
        try:
            res     = requests.post(f"{BASE_URL}/api/query",
                                    json={"question": q},
                                    headers=headers,
                                    timeout=600)
            elapsed = time.time() - start
            status  = res.status_code
            score   = res.json().get("confidence_score", 0) if res.ok else 0
        except Exception as e:
            elapsed = time.time() - start
            status  = 0
            score   = 0
            print(f"  Error: {e}")

        results.append(elapsed)
        print(f" [{status}] {q[:45]:<45} → {elapsed:.1f}s  confidence={score:.2f}")

    print(f"\n{'─'*60}")
    print(f"  Queries:   {len(results)}")
    print(f"  Avg time:  {statistics.mean(results):.1f}s")
    print(f"  Min time:  {min(results):.1f}s")
    print(f"  Max time:  {max(results):.1f}s")
    print(f"  Median:    {statistics.median(results):.1f}s")
    print(f"{'─'*60}")


if __name__ == "__main__":
    run_benchmark()
