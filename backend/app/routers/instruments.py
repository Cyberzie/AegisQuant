from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.dependencies import get_db
from app.models.instrument import Instrument
from app.schemas.instrument import (
    InstrumentCreate,
    InstrumentResponse,
    InstrumentUpdate,
)


router = APIRouter(
    prefix="/instruments",
)


@router.post(
    "/",
    response_model=InstrumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_instrument(
    instrument: InstrumentCreate,
    db: Session = Depends(get_db),
):
    existing_instrument = (
        db.query(Instrument)
        .filter(Instrument.symbol == instrument.symbol)
        .first()
    )

    if existing_instrument:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Instrument with this symbol already exists.",
        )

    new_instrument = Instrument(
        symbol=instrument.symbol,
        name=instrument.name,
        asset_type=instrument.asset_type,
        exchange=instrument.exchange,
        currency=instrument.currency,
    )

    db.add(new_instrument)
    db.commit()
    db.refresh(new_instrument)

    return new_instrument


@router.get(
    "/",
    response_model=list[InstrumentResponse],
)
def list_instruments(
    db: Session = Depends(get_db),
):
    return (
        db.query(Instrument)
        .order_by(Instrument.id)
        .all()
    )


@router.get(
    "/{instrument_id}",
    response_model=InstrumentResponse,
)
def get_instrument(
    instrument_id: int,
    db: Session = Depends(get_db),
):
    instrument = (
        db.query(Instrument)
        .filter(Instrument.id == instrument_id)
        .first()
    )

    if instrument is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instrument not found.",
        )

    return instrument


@router.patch(
    "/{instrument_id}",
    response_model=InstrumentResponse,
)
def update_instrument(
    instrument_id: int,
    instrument: InstrumentUpdate,
    db: Session = Depends(get_db),
):
    existing_instrument = (
        db.query(Instrument)
        .filter(Instrument.id == instrument_id)
        .first()
    )

    if existing_instrument is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instrument not found.",
        )

    update_data = instrument.model_dump(
        exclude_unset=True,
    )

    for field, value in update_data.items():
        setattr(existing_instrument, field, value)

    db.commit()
    db.refresh(existing_instrument)

    return existing_instrument
