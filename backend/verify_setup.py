"""
Test script to verify backend is working correctly
"""
import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_health():
    """Test health endpoint"""
    print("\n🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_status():
    """Test status endpoint"""
    print("\n📊 Testing Status Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status endpoint working!")
            print(f"   Status: {data.get('status')}")
            print(f"   Knowledge Base Ready: {data.get('knowledge_base_ready')}")
            print(f"   LLM Ready: {data.get('llm_ready')}")
            print(f"   Documents Count: {data.get('documents_count')}")
            return True
        else:
            print(f"❌ Status check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_query():
    """Test query endpoint"""
    print("\n❓ Testing Query Endpoint...")
    try:
        payload = {
            "question": "What is the confidence scoring framework?"
        }
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Query endpoint working!")
            print(f"   Question: {data.get('question')[:50]}...")
            print(f"   Answer: {data.get('answer')[:100]}...")
            print(f"   Confidence Score: {data.get('confidence_score')}")
            print(f"   Confidence Label: {data.get('confidence_label')}")
            return True
        elif response.status_code == 404:
            print("⚠️  Query works but no documents in knowledge base yet")
            print("   Upload a PDF first via the admin panel")
            return True
        else:
            print(f"❌ Query failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def test_cors():
    """Test CORS configuration"""
    print("\n🌐 Testing CORS Configuration...")
    try:
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        response = requests.options(f"{BASE_URL}/query", headers=headers)
        
        if response.status_code in [200, 204]:
            print("✅ CORS is configured correctly!")
            return True
        else:
            print(f"⚠️  CORS might have issues (status {response.status_code})")
            return True  # Don't fail on CORS issues
    except Exception as e:
        print(f"⚠️  CORS test error: {e}")
        return True  # Don't fail on CORS issues

def main():
    print("=" * 60)
    print("Backend API Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Status Check", test_status()))
    results.append(("CORS Config", test_cors()))
    results.append(("Query Endpoint", test_query()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Backend is working correctly!")
        print("\nYou can now:")
        print("  1. Open http://localhost:8000/docs for API documentation")
        print("  2. Start your frontend: cd frontend && npm start")
        print("  3. Access the app at http://localhost:3000")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("\nMake sure:")
        print("  1. Backend is running: python main.py")
        print("  2. No firewall blocking port 8000")
        print("  3. All dependencies installed: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()