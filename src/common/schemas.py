from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

EventType = Literal["view", "addtocart", "transaction"]


class BronzeEvent(BaseModel):
    """Raw record as ingested from events.csv, with minimal typing."""

    event_time: datetime
    visitorid: int
    event: str 
    itemid: int
    transactionid: Optional[int] = None

    model_config = {"populate_by_name": True}


class SilverEvent(BaseModel):
    """Cleaned and validated record."""

    event_time: datetime
    event_date: date
    visitorid: int
    event: EventType
    itemid: int
    transactionid: Optional[int] = None

    @model_validator(mode="after")
    def transaction_id_matches_event_type(self) -> "SilverEvent":
        has_txn_id = self.transactionid is not None
        is_transaction = self.event == "transaction"
        if has_txn_id != is_transaction:
            raise ValueError(
                "transactionid must be set if and only if event == 'transaction'"
            )
        return self


class GoldFeatures(BaseModel):
    """Feature-ready snapshot record consumed by inference."""

    visitorid: int
    snapshot_date: date
    user_views_7d: int
    user_addtocart_7d: int
    user_transactions_30d: int
    user_sessions_30d: int
    avg_session_duration_30d: float
    user_cart_items_avg_popularity_7d: float
    user_cart_top_seller_ratio_7d: float