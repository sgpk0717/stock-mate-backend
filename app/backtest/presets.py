"""내장 프리셋 전략."""

from .schemas import (
    ConditionSchema,
    PositionSizingSchema,
    RiskManagementSchema,
    ScalingSchema,
    StrategyInfo,
    StrategySchema,
)

PRESETS: list[StrategyInfo] = [
    StrategyInfo(
        name="rsi_oversold",
        description="RSI 과매도 반등 매수 / 과매수 매도",
        strategy=StrategySchema(
            name="RSI 과매도 반등",
            description="RSI가 30 이하로 떨어지면 매수, 70 이상이면 매도하는 전략",
            timeframe="1d",
            buy_conditions=[
                ConditionSchema(indicator="rsi", params={"period": 14}, op="<=", value=30),
            ],
            buy_logic="AND",
            sell_conditions=[
                ConditionSchema(indicator="rsi", params={"period": 14}, op=">=", value=70),
            ],
            sell_logic="OR",
        ),
    ),
    StrategyInfo(
        name="golden_cross",
        description="이동평균 골든크로스 매수 / 데드크로스 매도",
        strategy=StrategySchema(
            name="골든크로스",
            description="단기 이동평균(5일)이 장기 이동평균(20일)을 상향 돌파하면 매수, 하향 돌파하면 매도",
            timeframe="1d",
            buy_conditions=[
                ConditionSchema(
                    indicator="golden_cross",
                    params={"fast_period": 5, "slow_period": 20},
                    op="==",
                    value=1,
                ),
            ],
            buy_logic="AND",
            sell_conditions=[
                ConditionSchema(
                    indicator="dead_cross",
                    params={"fast_period": 5, "slow_period": 20},
                    op="==",
                    value=1,
                ),
            ],
            sell_logic="OR",
        ),
    ),
    StrategyInfo(
        name="macd_cross",
        description="MACD 시그널 크로스 매수/매도",
        strategy=StrategySchema(
            name="MACD 크로스",
            description="MACD가 시그널선을 상향 돌파하면 매수, 하향 돌파하면 매도",
            timeframe="1d",
            buy_conditions=[
                ConditionSchema(
                    indicator="macd_cross",
                    params={"fast": 12, "slow": 26, "signal": 9},
                    op="==",
                    value=1,
                ),
            ],
            buy_logic="AND",
            sell_conditions=[
                ConditionSchema(
                    indicator="macd_hist",
                    params={"fast": 12, "slow": 26, "signal": 9},
                    op="<",
                    value=0,
                ),
            ],
            sell_logic="OR",
        ),
    ),
    StrategyInfo(
        name="rsi_conviction_scaling",
        description="RSI 확신도 기반 분할매수/매도 + 트레일링 스탑",
        strategy=StrategySchema(
            name="RSI 확신도 + 분할매매",
            description="RSI 과매도에서 확신도 기반 분할진입, 트레일링 스탑으로 수익 극대화",
            timeframe="1d",
            buy_conditions=[
                ConditionSchema(indicator="rsi", params={"period": 14}, op="<=", value=30),
                ConditionSchema(indicator="volume_ratio", params={"period": 20}, op=">=", value=1.5),
            ],
            buy_logic="AND",
            sell_conditions=[
                ConditionSchema(indicator="rsi", params={"period": 14}, op=">=", value=70),
            ],
            sell_logic="OR",
            position_sizing=PositionSizingSchema(
                mode="conviction",
                weights={"rsi": 0.6, "volume_ratio": 0.4},
            ),
            scaling=ScalingSchema(
                enabled=True,
                initial_pct=0.5,
                scale_in_drop_pct=3.0,
                partial_exit_pct=0.5,
                partial_exit_gain_pct=5.0,
            ),
            risk_management=RiskManagementSchema(
                stop_loss_pct=5.0,
                trailing_stop_pct=2.0,
            ),
        ),
    ),
]

PRESET_MAP = {p.name: p for p in PRESETS}
