from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
from pathlib import Path
import os

# --- PATH CONFIGURATION ---
# Get the absolute path of the 'backend' folder
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"

# Create the data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Set the database URL
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
    role = Column(String, default="user")  # 'admin' or 'user'
    is_active = Column(Boolean, default=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    chunk_count = Column(Integer)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(String)
    answer = Column(String)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship to Feedback
    feedback = relationship("Feedback", back_populates="chat_history", uselist=False)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    chat_history_id = Column(Integer, ForeignKey("chat_history.id"))
    rating = Column(Integer)  # 1-5 stars
    comment = Column(String, nullable=True) # Reason if rating < 5
    
    chat_history = relationship("ChatHistory", back_populates="feedback")

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()