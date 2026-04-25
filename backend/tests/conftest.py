"""
Test configuration — patches out heavy ML services so pytest
never triggers a model download or ChromaDB init.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from core.database import Base, get_db, User, Session as ChatSession
from core.security import get_current_user

# ---------------------------------------------------------------------------
# In-memory test database
# ---------------------------------------------------------------------------

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(name="client")
def fixture_client():
    # ✅ Create all tables before every test that uses client
    Base.metadata.create_all(bind=engine)
    yield TestClient(app)
    Base.metadata.drop_all(bind=engine)

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
    user = User(email="tester@example.com", hashed_password="fake", role="user")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    app.dependency_overrides[get_current_user] = lambda: user
    yield user
    del app.dependency_overrides[get_current_user]

@pytest.fixture(name="test_session")
def fixture_test_session(db_session, test_user):
    session = ChatSession(user_id=test_user.id, title="Test Session")
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session

# ---------------------------------------------------------------------------
# Auto-mock heavy ML services — applied to EVERY test automatically
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_heavy_services():
    mock_llama   = MagicMock()
    mock_chroma  = MagicMock()
    mock_scoring = MagicMock()

    mock_llama.is_ready.return_value   = True
    mock_chroma.is_ready.return_value  = True
    mock_chroma.get_count.return_value = 0
    mock_chroma.search.return_value    = []
    mock_llama.generate_answer.return_value = "Mock answer"
    mock_scoring.compute_confidence_score.return_value = (0.9, "Mock explanation", [], {})

    with patch("api.routes.endpoints.llama_service",   mock_llama), \
         patch("api.routes.endpoints.chroma_service",  mock_chroma), \
         patch("api.routes.endpoints.scoring_service", mock_scoring):
        yield