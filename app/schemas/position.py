from pydantic import BaseModel


class PositionResponse(BaseModel):
    id: int
    symbol: str
    name: str
    mode: str
    qty: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_percent: float
