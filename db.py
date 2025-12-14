# db.py
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()
engine = create_engine("sqlite:///cedropass.db")
SessionLocal = sessionmaker(bind=engine)

def gen_id():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=gen_id)
    created_at = Column(DateTime, default=datetime.utcnow)


class Chat(Base):
    __tablename__ = "chats"
    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, ForeignKey("users.id"))
    project_name = Column(String)
    learning_enabled = Column(Boolean, default=True)  # ✅ opt-in learning
    created_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=gen_id)
    chat_id = Column(String, ForeignKey("chats.id"))
    role = Column(String)  # user / assistant / user_feedback
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)
