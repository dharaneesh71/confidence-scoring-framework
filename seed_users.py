"""
seed_users.py — First-deploy admin seeder
------------------------------------------
Run this ONCE after the backend is live to create the initial admin
and any default test users. After that, users register through the UI.

Usage:
    python seed_users.py
    python seed_users.py --url http://localhost:8000   # custom backend URL
"""
import argparse
import requests

def create_user(api_url: str, email: str, password: str, label: str):
    print(f"Creating {label} ({email})...")
    try:
        response = requests.post(
            f"{api_url}/api/auth/register",
            json={"email": email, "password": password},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Created  — ID: {data['id']} | Role: {data['role']}")
        elif response.status_code == 400:
            print(f"  ⚠️  Already exists: {email}")
        else:
            print(f"  ❌ Failed ({response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Could not connect to backend at {api_url}. Is it running?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed initial users into CONFID.AI")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    args = parser.parse_args()

    print(f"\nSeeding users into {args.url}\n{'─' * 40}")

    # The FIRST user registered automatically becomes admin (see endpoints.py)
    create_user(args.url, "admin@example.com", "admin123", "ADMIN")

    # Regular users
    create_user(args.url, "user@example.com",  "user123",  "NORMAL USER")
    create_user(args.url, "user1@example.com", "user1234", "NORMAL USER")

    print("\nDone. Users can now self-register at the /login page.")