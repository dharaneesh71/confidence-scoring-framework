from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
import datetime
from pathlib import Path

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR}/confid_ai.db"

# --- DATABASE SETUP ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    is_active = Column(Boolean, default=True)

    sessions = relationship("Session", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    question = Column(String)
    answer = Column(String)
    confidence_score = Column(Float)
    explanation = Column(Text, nullable=True)
    citations = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="chat_history")
    session = relationship("Session", back_populates="messages")
    feedback = relationship("Feedback", back_populates="chat_history", uselist=False)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    chunk_count = Column(Integer)
    domain = Column(String, default="General", nullable=True)  

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history.id"))
    rating = Column(Integer)
    comment = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    chat_history = relationship("ChatHistory", back_populates="feedback")

# --- AUTO-MIGRATE: add domain column if it doesn't exist (safe for existing DBs) ---
def _migrate_add_domain_column():
    """
    SQLite doesn't support ALTER TABLE ADD COLUMN if it already exists.
    This safely adds the domain column to an existing database without
    wiping any data.
    """
    import sqlite3
    db_path = DATA_DIR / "confid_ai.db"
    if not db_path.exists():
        return  # fresh DB — create_all handles it
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if column already exists
    cursor.execute("PRAGMA table_info(documents)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    
    if "domain" not in existing_columns:
        cursor.execute("ALTER TABLE documents ADD COLUMN domain VARCHAR DEFAULT 'General'")
        conn.commit()
        print("[DB Migration] ✅ Added 'domain' column to documents table.")
    
    conn.close()

_migrate_add_domain_column() 
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
