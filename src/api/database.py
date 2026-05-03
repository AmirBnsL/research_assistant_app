import os
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./data/assistant.db"

os.makedirs("./data", exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String)
    session_id = Column(String, nullable=True, index=True)


class Evidence(Base):
    __tablename__ = "evidence"
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, index=True)
    session_id = Column(String, index=True)  # The main topic tracker
    source = Column(String)
    title = Column(String)
    authors = Column(String)
    score = Column(Float)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
