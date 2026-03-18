"""한국 주식 호가 단위 (tick size) 유틸리티.

KRX 호가 단위 규정에 따라 가격을 유효 호가 단위로 변환한다.
"""

# (하한가격, 호가 단위) — 가격이 threshold 이상일 때 해당 tick size 적용
_TICK_TABLE: list[tuple[int, int]] = [
    (0, 1),
    (2_000, 5),
    (5_000, 10),
    (20_000, 50),
    (50_000, 100),
    (200_000, 500),
    (500_000, 1_000),
]


def get_tick_size(price: int) -> int:
    """주어진 가격의 호가 단위 반환.

    >>> get_tick_size(1500)
    1
    >>> get_tick_size(3000)
    5
    >>> get_tick_size(70000)
    100
    """
    tick = 1
    for threshold, size in _TICK_TABLE:
        if price >= threshold:
            tick = size
        else:
            break
    return tick


def round_to_tick(price: int | float, direction: str = "nearest") -> int:
    """가격을 유효 호가 단위로 반올림.

    Args:
        price: 원래 가격.
        direction: "nearest" (반올림) | "down" (내림) | "up" (올림).

    Returns:
        호가 단위에 맞는 정수 가격.

    >>> round_to_tick(70530, "down")
    70500
    >>> round_to_tick(70530, "up")
    70600
    >>> round_to_tick(70550)
    70600
    """
    p = int(price)
    if p <= 0:
        return 0
    tick = get_tick_size(p)
    if direction == "down":
        return (p // tick) * tick
    elif direction == "up":
        return ((p + tick - 1) // tick) * tick
    else:  # nearest
        return ((p + tick // 2) // tick) * tick


def tick_down(price: int, n: int = 1) -> int:
    """현재 가격에서 n틱 하락한 가격.

    >>> tick_down(70500, 1)
    70400
    >>> tick_down(5000, 1)
    4995
    """
    p = price
    for _ in range(n):
        tick = get_tick_size(p)
        p -= tick
    return max(p, 0)


def tick_up(price: int, n: int = 1) -> int:
    """현재 가격에서 n틱 상승한 가격.

    >>> tick_up(70500, 1)
    70600
    """
    p = price
    for _ in range(n):
        tick = get_tick_size(p)
        p += tick
    return p


def is_valid_tick(price: int) -> bool:
    """가격이 호가 단위에 맞는지 확인.

    >>> is_valid_tick(70500)
    True
    >>> is_valid_tick(70530)
    False
    """
    if price <= 0:
        return False
    tick = get_tick_size(price)
    return price % tick == 0
