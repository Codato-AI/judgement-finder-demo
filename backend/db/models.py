from datetime import datetime
from sqlalchemy import String, Text, DateTime, Float, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column
from db.database import Base


class CaseLaw(Base):
    __tablename__ = "case_laws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    original_case_no: Mapped[str | None] = mapped_column(String(255), nullable=True)
    judgment_date: Mapped[str | None] = mapped_column(String(50), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(500), nullable=True)
    acts_cited: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    anonymized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    pii_map: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    has_embedding: Mapped[bool] = mapped_column(default=False)
    faiss_index: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ModelTestResult(Base):
    __tablename__ = "model_test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[int] = mapped_column(Integer, index=True)
    model_id: Mapped[str] = mapped_column(String(100))
    model_label: Mapped[str] = mapped_column(String(100))
    generated_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    score_completeness: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_pii_safety: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_readability: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_structure: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_legal_terms: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_total: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
