"""알파 마이닝 DB 모델."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class AlphaMiningRun(Base):
    __tablename__ = "alpha_mining_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="'PENDING'"
    )
    progress: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    factors_found: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    total_evaluated: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    iteration_logs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AlphaFactor(Base):
    __tablename__ = "alpha_factors"
    __table_args__ = (
        Index("ix_alpha_factors_mining_run_id", "mining_run_id"),
        Index("ix_alpha_factors_status", "status"),
        Index("ix_alpha_factors_ic_mean", "ic_mean"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    mining_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("alpha_mining_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    expression_str: Mapped[str] = mapped_column(Text, nullable=False)
    expression_sympy: Mapped[str | None] = mapped_column(Text, nullable=True)
    polars_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    # IC 메트릭
    ic_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    ic_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    icir: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 상태
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="'discovered'"
    )

    # Phase 2: 인과 검증
    causal_robust: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    causal_effect_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    causal_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Evolution Engine
    fitness_composite: Mapped[float | None] = mapped_column(Float, nullable=True)
    tree_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tree_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expression_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    operator_origin: Mapped[str | None] = mapped_column(String(30), nullable=True)
    is_elite: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    population_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    birth_generation: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    # Phase 3: 계보 + 복합 팩터
    parent_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    factor_type: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="'single'"
    )
    component_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AlphaExperience(Base):
    """벡터 임베딩 기반 경험 메모리."""

    __tablename__ = "alpha_experiences"
    __table_args__ = (
        Index("ix_alpha_experiences_success", "success"),
        Index("ix_alpha_experiences_ic_mean", "ic_mean"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    factor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("alpha_factors.id", ondelete="SET NULL"),
        nullable=True,
    )
    expression_str: Mapped[str] = mapped_column(Text, nullable=False)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ic_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    generation: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    parent_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
