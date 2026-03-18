"""SymPy AST → 한글 자연어 번역기.

알파 팩터 수식을 사람이 읽을 수 있는 한글 설명으로 변환한다.

SymPy 내부 변환 주의:
- a / b  →  Mul(a, Pow(b, -1))
- a - b  →  Add(a, Mul(-1, b))
이 패턴들을 자연스러운 ÷, - 표기로 복원한다.
"""

from __future__ import annotations

import sympy

from app.alpha.ast_converter import Clip

# ── 변수 → 한글 이름 매핑 ──

_KOREAN_NAMES: dict[str, str] = {
    # OHLCV 기본
    "close": "종가",
    "open": "시가",
    "high": "고가",
    "low": "저가",
    "volume": "거래량",
    # 기존 지표
    "sma_20": "20일 이동평균",
    "sma": "20일 이동평균",
    "ema_20": "20일 지수이동평균",
    "rsi": "RSI(상대강도지수)",
    "volume_ratio": "거래량비율",
    "vol_ratio": "거래량비율",
    "atr_14": "ATR(14일 평균진폭)",
    "atr": "ATR(14일 평균진폭)",
    "macd_hist": "MACD 히스토그램",
    "macd": "MACD 히스토그램",
    "bb_upper": "볼린저밴드 상단",
    "bb_lower": "볼린저밴드 하단",
    "bb_width": "볼린저밴드 폭",
    "price_change_pct": "전일대비 등락률",
    "pct_change": "전일대비 등락률",
    # 멀티 윈도우 이동평균
    "sma_5": "5일 이동평균",
    "sma_10": "10일 이동평균",
    "sma_60": "60일 이동평균",
    "ema_5": "5일 지수이동평균",
    "ema_10": "10일 지수이동평균",
    "ema_60": "60일 지수이동평균",
    # 멀티 윈도우 RSI/ATR
    "rsi_7": "RSI(7일)",
    "rsi_21": "RSI(21일)",
    "atr_7": "ATR(7일 평균진폭)",
    "atr_21": "ATR(21일 평균진폭)",
    # 시차 피처
    "close_lag_1": "전일 종가",
    "close_lag_5": "5일전 종가",
    "close_lag_20": "20일전 종가",
    "volume_lag_1": "전일 거래량",
    "volume_lag_5": "5일전 거래량",
    # N일 수익률
    "return_5d": "5일 수익률",
    "return_20d": "20일 수익률",
    # 파생 피처
    "bb_position": "볼린저밴드 위치(0~1)",
    # 횡단면 피처
    "rank_close": "종가 횡단면 순위",
    "rank_volume": "거래량 횡단면 순위",
    "zscore_close": "종가 횡단면 Z-score",
    "zscore_volume": "거래량 횡단면 Z-score",
    # 투자자 수급
    "foreign_net_norm": "외국인 순매수 강도",
    "inst_net_norm": "기관 순매수 강도",
    "retail_net_norm": "개인 순매수 강도",
    "foreign_buy_ratio": "외국인 매수비율",
    "inst_buy_ratio": "기관 매수비율",
    "retail_buy_ratio": "개인 매수비율",
    # DART 재무
    "eps": "주당순이익(EPS)",
    "bps": "주당순자산(BPS)",
    "operating_margin": "영업이익률",
    "debt_to_equity": "부채비율",
    "earnings_yield": "이익수익률",
    "book_yield": "자산수익률",
    # 뉴스 감성
    "sentiment_score": "뉴스 감성 점수",
    "event_score": "이벤트 스코어",
    # 섹터
    "sector_return": "섹터 평균수익률",
    "sector_rel_strength": "섹터 상대강도",
    "sector_rank": "섹터 내 순위",
    # 신용/공매도
    "margin_rate": "융자잔고율",
    "short_balance_rate": "대차잔고비율",
    "short_volume_ratio": "공매도 비율",
    # 프로그램 매매
    "pgm_net_norm": "프로그램 순매수 강도",
    "pgm_buy_ratio": "프로그램 매수비율",
    # ── 이벤트 감지 피처 ──
    "vol_spike_5d": "5일 거래량 폭증비",
    "vol_spike_20d": "20일 거래량 폭증비",
    "consec_low_vol_5": "5일중 저거래량 일수",
    "vol_dry_then_spike": "거래량 고갈 후 폭발",
    "consec_up": "5일중 상승일 수",
    "consec_down": "5일중 하락일 수",
    "gap_up_pct": "갭상승률",
    "gap_down_pct": "갭하락률",
    "range_breakout": "20일 신고가 돌파",
    "range_breakdown": "20일 신저가 이탈",
    "rsi_oversold_bounce": "RSI 과매도 반등",
    "macd_cross_up": "MACD 골든크로스",
    "bb_squeeze": "볼린저밴드 수축",
    "bb_breakout_upper": "BB 상단 돌파",
    "foreign_accumulate_5d": "외국인 5일 연속매수",
    "inst_accumulate_5d": "기관 5일 연속매수",
}

# ── 연산자 origin → 한글 근거 ──

_OPERATOR_REASONS: dict[str, str] = {
    "mutate": "무작위 연산자 변이(Add↔Mul, 상수 섭동, 피처 교체 등)로 생성",
    "ast_mutate_operator": "무작위 연산자 변이(Add↔Mul 교체)로 생성",
    "ast_mutate_constant": "기존 수식의 상수를 무작위 섭동(±20%)하여 생성",
    "ast_mutate_feature": "기존 수식의 피처를 다른 피처로 무작위 교체하여 생성",
    "ast_mutate_function": "함수 래핑(log, sqrt 등) 또는 제거로 생성",
    "crossover": "두 부모 수식의 서브트리를 교차하여 생성",
    "ast_crossover": "두 부모 수식의 서브트리를 교차하여 생성",
    "hoist": "기존 수식에서 의미 있는 서브트리만 추출하여 단순화",
    "ast_hoist": "기존 수식에서 의미 있는 서브트리만 추출하여 단순화",
    "ephemeral_constant": "기존 수식의 변수를 무작위 상수로 교체하여 생성",
    "ast_ephemeral_constant": "기존 수식의 변수를 무작위 상수(0.01~100)로 교체하여 생성",
    "llm_seed": "AI(Claude)가 새로 설계한 수식",
    "llm_mutate": "AI(Claude)가 기존 수식의 IC(정보계수)를 개선하도록 변형",
    "initial": "초기 모집단 시드 수식",
    "seed": "초기 모집단 시드 수식",
}


def _format_number(val: float) -> str:
    """숫자를 간결한 문자열로."""
    if val == int(val):
        return str(int(val))
    return f"{val:.4g}"


def _is_negative_one(expr: sympy.Basic) -> bool:
    """expr이 -1인지 확인."""
    return expr == sympy.S.NegativeOne


def _is_reciprocal(expr: sympy.Basic) -> tuple[bool, sympy.Basic | None]:
    """Pow(x, -1) 패턴을 감지하여 (True, x) 반환."""
    if isinstance(expr, sympy.Pow):
        if expr.args[1] == sympy.S.NegativeOne:
            return True, expr.args[0]
    return False, None


def sympy_to_korean(expr: sympy.Basic, depth: int = 0) -> str:
    """SymPy 표현식을 한글 자연어로 번역."""
    if depth > 5:
        return "(...)"

    # Symbol → 한글 이름
    if isinstance(expr, sympy.Symbol):
        name = str(expr)
        return _KOREAN_NAMES.get(name, name)

    # 숫자 리터럴
    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        return _format_number(float(expr))
    if isinstance(expr, (int, float)):
        return _format_number(float(expr))
    if expr.is_number:
        return _format_number(float(expr))

    # log(x)
    if isinstance(expr, sympy.log):
        inner = sympy_to_korean(expr.args[0], depth + 1)
        return f"log({inner})"

    # exp(x)
    if isinstance(expr, sympy.exp):
        inner = sympy_to_korean(expr.args[0], depth + 1)
        return f"exp({inner})"

    # |x|
    if isinstance(expr, sympy.Abs):
        inner = sympy_to_korean(expr.args[0], depth + 1)
        return f"|{inner}|"

    # sign(x) → 부호
    if isinstance(expr, sympy.sign):
        inner = sympy_to_korean(expr.args[0], depth + 1)
        return f"부호({inner})"

    # Heaviside(x) → 초과시1
    if isinstance(expr, sympy.Heaviside):
        inner = sympy_to_korean(expr.args[0], depth + 1)
        return f"양수면1({inner})"

    # Max(x, y)
    if isinstance(expr, sympy.Max):
        parts = [sympy_to_korean(a, depth + 1) for a in expr.args]
        return f"큰값({', '.join(parts)})"

    # Min(x, y)
    if isinstance(expr, sympy.Min):
        parts = [sympy_to_korean(a, depth + 1) for a in expr.args]
        return f"작은값({', '.join(parts)})"

    # clip(x, lo, hi)
    if isinstance(expr, Clip):
        x = sympy_to_korean(expr.args[0], depth + 1)
        lo = sympy_to_korean(expr.args[1], depth + 1)
        hi = sympy_to_korean(expr.args[2], depth + 1)
        return f"범위제한({x}, {lo}~{hi})"

    # x^n — sqrt 및 나눗셈 패턴 처리
    if isinstance(expr, sympy.Pow):
        base_str = sympy_to_korean(expr.args[0], depth + 1)
        exp_val = expr.args[1]

        # sqrt: x^(1/2)
        if exp_val == sympy.Rational(1, 2) or exp_val == sympy.Float(0.5):
            return f"√({base_str})"

        # 역수: x^(-1) → 단독으로 올 때는 "1/{x}"
        if exp_val == sympy.S.NegativeOne:
            return f"(1 ÷ {base_str})"

        if isinstance(exp_val, (sympy.Integer, sympy.Float, sympy.Rational)):
            return f"({base_str})^{_format_number(float(exp_val))}"

        exp_str = sympy_to_korean(exp_val, depth + 1)
        return f"({base_str})^({exp_str})"

    # Mul: a * b * ... — 나눗셈(Mul(a, Pow(b,-1))) 복원
    if isinstance(expr, sympy.Mul):
        return _translate_mul(expr, depth)

    # Add: a + b + ... — 뺄셈(Add(a, Mul(-1,b))) 복원
    if isinstance(expr, sympy.Add):
        return _translate_add(expr, depth)

    # 기타
    return str(expr)


def _translate_mul(expr: sympy.Mul, depth: int) -> str:
    """Mul 노드를 나눗셈 패턴까지 인식하여 번역."""
    args = list(expr.args)

    # 음수 부호: Mul(-1, x, ...) → -(x × ...)
    negative = False
    if args and _is_negative_one(args[0]):
        negative = True
        args = args[1:]

    numerator: list[str] = []
    denominator: list[str] = []

    for arg in args:
        is_recip, base = _is_reciprocal(arg)
        if is_recip and base is not None:
            # Pow(x, -1) → 분모
            base_str = sympy_to_korean(base, depth + 1)
            if isinstance(base, sympy.Add):
                base_str = f"({base_str})"
            denominator.append(base_str)
        else:
            part = sympy_to_korean(arg, depth + 1)
            if isinstance(arg, sympy.Add):
                part = f"({part})"
            numerator.append(part)

    # 조합
    if not numerator:
        numer_str = "1"
    elif len(numerator) == 1:
        numer_str = numerator[0]
    else:
        numer_str = " × ".join(numerator)

    if denominator:
        if len(denominator) == 1:
            denom_str = denominator[0]
        else:
            denom_str = " × ".join(denominator)
        result = f"{numer_str} ÷ {denom_str}"
    else:
        result = numer_str

    if negative:
        result = f"-({result})" if (" × " in result or " ÷ " in result) else f"-{result}"

    return result


def _translate_add(expr: sympy.Add, depth: int) -> str:
    """Add 노드를 뺄셈 패턴까지 인식하여 번역."""
    positive: list[str] = []
    negative: list[str] = []

    for arg in expr.args:
        # Mul(-1, x) → 뺄셈 항
        if isinstance(arg, sympy.Mul):
            mul_args = list(arg.args)
            if mul_args and _is_negative_one(mul_args[0]):
                # -1 * x → x를 빼기
                rest = mul_args[1:]
                if len(rest) == 1:
                    neg_part = sympy_to_korean(rest[0], depth + 1)
                else:
                    inner_mul = sympy.Mul(*rest)
                    neg_part = sympy_to_korean(inner_mul, depth + 1)
                negative.append(neg_part)
                continue

        # 음수 상수
        if arg.is_number and float(arg) < 0:
            negative.append(_format_number(abs(float(arg))))
            continue

        positive.append(sympy_to_korean(arg, depth + 1))

    # 조합: 양수항들 + 음수항들
    if not positive:
        if not negative:
            return "0"
        result = f"-{negative[0]}"
        for n in negative[1:]:
            result += f" - {n}"
    else:
        result = positive[0]
        for p in positive[1:]:
            result += f" + {p}"
        for n in negative:
            result += f" - {n}"

    return result


def generate_hypothesis_korean(
    expr: sympy.Basic,
    operator_origin: str,
) -> str:
    """수식의 한글 설명 + 변이 근거를 합친 최종 설명문."""
    korean_formula = sympy_to_korean(expr)
    reason = _OPERATOR_REASONS.get(
        operator_origin, _OPERATOR_REASONS.get("initial", "")
    )

    return f"{korean_formula}\n\n[생성 방식] {reason}"
