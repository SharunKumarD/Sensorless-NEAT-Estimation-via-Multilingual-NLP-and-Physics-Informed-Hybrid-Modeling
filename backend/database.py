from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./neat_tracker.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, index=True)
    height = Column(Float)
    weight = Column(Float)
    age = Column(Integer)
    gender = Column(String)

class WorkoutHistory(Base):
    __tablename__ = "workout_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    extracted_met = Column(Float)
    duration = Column(Float)
    live_temp = Column(Float)
    calories_burned = Column(Float)
    shap_array = Column(JSON)
