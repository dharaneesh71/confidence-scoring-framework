import requests

API_URL = "http://localhost:8000/api/auth/register"

def create_user(email, password, label):
    print(f"Creating {label} ({email})...")
    response = requests.post(API_URL, json={"email": email, "password": password})
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Created User ID: {data['id']} | Role: {data['role']}")
    elif response.status_code == 400:
        print(f"⚠️  Account already exists: {email}")
    else:
        print(f"❌ Failed: {response.text}")

if __name__ == "__main__":
    # 1. Create the Admin (Must be first!)
    create_user("admin@example.com", "admin123", "ADMIN")

    # 2. Create the Regular User
    create_user("user@example.com", "user123", "NORMAL USER")
    create_user("user1@example.com", "user1234", "NORMAL USER")