"""Read-only prediction-market adapters for N13 Stage 3."""

from .base import GameMarket, MarketRef, Quote
from .gap import GapResult, compute_gap
from .kalshi import KalshiClient
from .polymarket import PolymarketClient
from .service import MarketPollResult, MarketService

__all__ = (
    "GameMarket",
    "GapResult",
    "KalshiClient",
    "MarketRef",
    "MarketPollResult",
    "MarketService",
    "PolymarketClient",
    "Quote",
    "compute_gap",
)
