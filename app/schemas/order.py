from pydantic import BaseModel


class OrderCreate(BaseModel):
    symbol: str
    side: str
    type: str
    price: float | None = None
    qty: int
    mode: str


class OrderResponse(BaseModel):
    order_id: str
    symbol: str
    name: str
    side: str
    type: str
    price: float | None
    qty: int
    status: str
    mode: str
    created_at: str
