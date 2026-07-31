from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class MarketData(Base):
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"),
        nullable=False,
        index=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        index=True,
    )

    open: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    high: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    low: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    close: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    volume: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    __table_args__ = (
        Index(
            "ix_market_data_instrument_timestamp",
            "instrument_id",
            "timestamp",
        ),
    )