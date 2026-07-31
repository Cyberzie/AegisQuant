from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    symbol: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    asset_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    exchange: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    currency: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )