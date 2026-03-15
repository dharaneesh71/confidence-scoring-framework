from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
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
    
    # Relationships
    sessions = relationship("Session", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True) # <--- NEW LINK TO SESSION
    question = Column(String)
    answer = Column(String)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chat_history")
    session = relationship("Session", back_populates="messages") # <--- NEW
    feedback = relationship("Feedback", back_populates="chat_history", uselist=False)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    chunk_count = Column(Integer)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history.id"))
    rating = Column(Integer)  
    comment = Column(String, nullable=True) 
    
    # ADDED: Timestamp for the Admin Dashboard sorting
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship: Links back to the specific ChatHistory
    chat_history = relationship("ChatHistory", back_populates="feedback")

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()