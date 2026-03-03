from pydantic import BaseModel


class StockInfoResponse(BaseModel):
    symbol: str
    name: str
    market: str


class CandleResponse(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


class CandleWithIndicatorsResponse(BaseModel):
    candles: list[CandleResponse]
    indicators: dict | None = None


class TickResponse(BaseModel):
    time: int  # Unix timestamp (KST)
    price: float
    volume: int = 0
