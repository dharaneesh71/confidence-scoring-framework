import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from main import app
from core.database import Base, get_db, User, ChatHistory, Feedback
from core.security import get_current_user # <--- We will override this!

# --- SETUP IN-MEMORY DATABASE ---
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool 
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- GLOBAL DB OVERRIDE ---
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

# --- FIXTURES ---

@pytest.fixture(name="db_session")
def fixture_db_session():
    """Creates a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(name="test_user")
def fixture_test_user(db_session):
    """Creates a user and forces the API to treat them as logged in."""
    # 1. Create User in DB
    user = User(email="tester@example.com", hashed_password="fake", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    # 2. OVERRIDE THE AUTH DEPENDENCY
    # This tells FastAPI: "Whenever an endpoint asks for 'current_user', return this object."
    app.dependency_overrides[get_current_user] = lambda: user
    
    yield user
    
    # Cleanup override after test
    del app.dependency_overrides[get_current_user]

# --- TESTS ---

def test_submit_feedback(db_session, test_user):
    # 1. Create a dummy Chat History entry for this user
    chat_entry = ChatHistory(
        user_id=test_user.id,
        question="Test Question",
        answer="Test Answer",
        confidence_score=0.9
    )
    db_session.add(chat_entry)
    db_session.commit()
    db_session.refresh(chat_entry)
    
    # 2. Test Submitting Feedback
    payload = {
        "history_id": chat_entry.id,
        "rating": 5,
        "comment": "Great answer!"
    }
    
    # No headers needed! The API thinks we are already logged in.
    response = client.post("/api/feedback", json=payload)

    # 3. Assertions
    assert response.status_code == 200
    assert response.json()["message"] == "Feedback received successfully"

    # 4. Verify DB
    feedback_in_db = db_session.query(Feedback).filter(Feedback.chat_history_id == chat_entry.id).first()
    assert feedback_in_db is not None
    assert feedback_in_db.rating == 5

def test_feedback_invalid_id(test_user):
    # Test submitting for a chat ID that doesn't exist
    payload = {
        "history_id": 9999, 
        "rating": 1
    }
    response = client.post("/api/feedback", json=payload)
    
    assert response.status_code == 404