"""시뮬레이션 리플레이 — SSE 스트림.

합성/실제 캔들 + 프리셋/알파팩터/커스텀 전략으로 OrderManager 시뮬레이션.
용도: 주문 실행 로직 검증 + 알파 팩터 판단 과정 디버깅.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid as _uuid
from datetime import date, timedelta

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sim-replay", tags=["sim-replay"])


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


def _load_preset_strategy(preset: str, buy_rsi: float, sell_rsi: float) -> dict:
    """프리셋/커스텀 전략 로드."""
    if preset == "custom":
        return {
            "buy_conditions": [
                {"indicator": "rsi", "params": {"period": 14}, "op": "<=", "value": buy_rsi},
            ],
            "sell_conditions": [
                {"indicator": "rsi", "params": {"period": 14}, "op": ">=", "value": sell_rsi},
            ],
            "buy_logic": "AND",
            "sell_logic": "AND",
        }
    from app.backtest.presets import PRESET_MAP
    info = PRESET_MAP.get(preset)
    if not info:
        return _load_preset_strategy("custom", buy_rsi, sell_rsi)
    s = info.strategy
    return {
        "buy_conditions": [c.model_dump() for c in s.buy_conditions],
        "sell_conditions": [c.model_dump() for c in s.sell_conditions],
        "buy_logic": s.buy_logic,
        "sell_logic": s.sell_logic,
    }


def _describe_signal(row: dict, strategy: dict, signal: int, alpha_indicator: str = "") -> str:
    """시그널 판단 근거."""
    if signal == 0:
        return ""
    side = "매수" if signal == 1 else "매도"
    conditions = strategy.get("buy_conditions" if signal == 1 else "sell_conditions", [])
    parts = []
    for c in conditions:
        ind = c.get("indicator", "?")
        op = c.get("op", "?")
        val = c.get("value", "?")
        actual = row.get(ind)
        if actual is not None:
            parts.append(f"{ind}={actual:.3f} {op} {val}")
        else:
            parts.append(f"{ind} {op} {val}")
    logic = strategy.get("buy_logic" if signal == 1 else "sell_logic", "AND")
    return f"{f' {logic} '.join(parts)} → {side} 시그널"


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None


@router.get("/stream")
async def stream_replay(
    symbol: str = Query("005930"),
    speed: float = Query(0.3, ge=0.05, le=5.0),
    stop_loss_pct: float = Query(5.0),
    trailing_stop_pct: float = Query(3.0),
    # 전략: 프리셋/커스텀
    strategy_preset: str = Query("rsi_oversold"),
    buy_rsi: float = Query(35),
    sell_rsi: float = Query(65),
    # 전략: 알파 팩터
    factor_id: str = Query("", description="알파 팩터 UUID"),
    # 데이터 소스
    data_source: str = Query("synthetic"),
    # 합성 모드
    base_price: int = Query(50000),
    n_bars: int = Query(78),
    scenario: str = Query("normal"),
    seed: int = Query(42),
    start_bar: int = Query(0),
    # 실제 모드
    start_date: str = Query(""),
    end_date: str = Query(""),
    interval: str = Query("5m"),
) -> StreamingResponse:
    """SSE 스트림 시뮬레이션 리플레이."""

    async def event_generator():
        import polars as pl
        from app.backtest.engine import generate_signals
        from app.trading.sim_engine import MarketSimulator, SimulationRunner

        alpha_indicator = ""  # 알파 팩터 인디케이터명 (있으면)

        # ── 전략 결정 ──
        if factor_id:
            # 알파 팩터 전략
            try:
                from app.alpha.models import AlphaFactor
                from app.core.database import async_session
                async with async_session() as db:
                    factor = await db.get(AlphaFactor, _uuid.UUID(factor_id))
                if not factor:
                    yield _sse_event("error", {"message": f"팩터 {factor_id[:8]} 없음"})
                    return

                from app.alpha.backtest_bridge import register_alpha_factor
                alpha_indicator = register_alpha_factor(str(factor.id), factor.expression_str)

                strategy = {
                    "buy_conditions": [
                        {"indicator": alpha_indicator, "params": {}, "op": ">", "value": 0.7},
                    ],
                    "sell_conditions": [
                        {"indicator": alpha_indicator, "params": {}, "op": "<", "value": 0.3},
                    ],
                    "buy_logic": "AND",
                    "sell_logic": "AND",
                }
                # 알파 팩터는 실제 데이터 강제
                effective_data_source = "real"
                effective_interval = factor.interval or interval
            except Exception as e:
                yield _sse_event("error", {"message": f"팩터 로드 실패: {e}"})
                return
        else:
            strategy = _load_preset_strategy(strategy_preset, buy_rsi, sell_rsi)
            effective_data_source = data_source
            effective_interval = interval

        # ── 데이터 준비 ──
        if effective_data_source == "real":
            sd = _parse_date(start_date) or date.today()
            ed = _parse_date(end_date) or sd

            try:
                if factor_id:
                    from app.backtest.data_loader import load_enriched_candles
                    df_raw = await load_enriched_candles(
                        [symbol], sd, ed, effective_interval,
                    )
                else:
                    from app.backtest.data_loader import load_candles
                    df_raw = await load_candles([symbol], sd, ed, effective_interval)
            except Exception as e:
                yield _sse_event("error", {"message": f"캔들 로딩 실패: {e}"})
                return

            df_sym = df_raw.filter(pl.col("symbol") == symbol).sort("dt")
            if df_sym.is_empty():
                yield _sse_event("error", {
                    "message": f"{symbol} {sd}~{ed} {effective_interval} 데이터 없음",
                })
                return

            first_close = int(df_sym["close"][0])
            tick_size = 100 if first_close >= 50000 else 50 if first_close >= 20000 else 10
            market = MarketSimulator(first_close, tick_size=tick_size, seed=42)

            try:
                df = generate_signals(df_sym, strategy)
            except Exception as e:
                yield _sse_event("error", {"message": f"시그널 생성 실패: {e}"})
                return

            total_bars = df.height
        else:
            # 합성 데이터
            tick_size = 100 if base_price >= 50000 else 50 if base_price >= 20000 else 10
            market = MarketSimulator(base_price, tick_size=tick_size, seed=seed)

            if scenario == "flash_crash":
                prices = market.scenario_flash_crash(n_bars, crash_at=n_bars // 2)
            elif scenario == "gap_up":
                prices = market.scenario_gap_up(n_bars, gap_at=n_bars // 3)
            else:
                prices = market.scenario_normal(n_bars)

            candles_df = market.generate_candles(n_bars, prices)
            try:
                df = generate_signals(candles_df, strategy)
            except Exception as e:
                yield _sse_event("error", {"message": f"시그널 생성 실패: {e}"})
                return
            total_bars = n_bars

        rows = df.to_dicts()
        runner = SimulationRunner(market)

        # 시뮬용 TTL
        from app.core.config import settings
        settings.ORDER_BUY_TTL_SECONDS = speed * 3
        settings.ORDER_SELL_TTL_SECONDS = speed * 2

        # 시작 이벤트
        yield _sse_event("start", {
            "symbol": symbol,
            "data_source": effective_data_source,
            "total_bars": total_bars,
            "strategy_preset": strategy_preset if not factor_id else f"alpha:{factor_id[:8]}",
            "alpha_indicator": alpha_indicator,
            "speed": speed,
        })

        for i, row in enumerate(rows):
            if i < start_bar:
                continue

            price = int(row["close"])
            signal = row.get("signal", 0)
            signal_detail = _describe_signal(row, strategy, signal, alpha_indicator)

            # 캔들 이벤트
            candle_data = {
                "bar": i, "total": total_bars,
                "dt": str(row.get("dt", "")),
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": price,
                "volume": row.get("volume", 0),
            }
            # 지표값 (디버깅용)
            for k in ("rsi", "macd_hist", "sma_5", "sma_20", "volume_ratio"):
                v = row.get(k)
                if v is not None:
                    candle_data[k] = round(float(v), 2)
            # 알파 팩터 값 (핵심 디버깅)
            if alpha_indicator:
                av = row.get(alpha_indicator)
                if av is not None:
                    candle_data["alpha_value"] = round(float(av), 4)

            yield _sse_event("candle", candle_data)

            # 시뮬 1틱
            await runner.run_tick(
                i, symbol, price, signal, signal_detail,
                stop_loss_pct=stop_loss_pct,
                trailing_stop_pct=trailing_stop_pct,
            )

            for event in runner.get_new_events():
                yield _sse_event("decision", event.to_dict())

            yield _sse_event("state", {"bar": i, **runner.get_state()})
            await asyncio.sleep(speed)

        state = runner.get_state()
        yield _sse_event("done", {
            "total_bars": total_bars,
            "total_events": len(runner.event_log.events),
            **state,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/strategies")
async def list_strategies():
    from app.backtest.presets import PRESETS
    result = [
        {"id": p.name, "name": p.strategy.name, "description": p.description}
        for p in PRESETS
    ]
    result.append({"id": "custom", "name": "커스텀 (RSI)", "description": "RSI 임계값 직접 설정"})
    return result


@router.get("/scenarios")
async def list_scenarios():
    return [
        {"id": "normal", "name": "정상 시장", "description": "±0.5% 랜덤워크"},
        {"id": "flash_crash", "name": "급락", "description": "중간 지점에서 -5% 급락"},
        {"id": "gap_up", "name": "갭 상승", "description": "1/3 지점에서 +3% 갭"},
    ]


@router.post("/analyze")
async def analyze_replay(body: dict):
    """시뮬레이션 결과 LLM 분석.

    판단 로그 전체를 Claude에 보내서 매매 흐름/주문 실행/리스크/개선점을 분석.
    """
    from app.core.llm._anthropic import chat_simple

    events = body.get("events", [])
    final_state = body.get("final_state", {})

    if not events:
        return {"analysis": "분석할 이벤트가 없습니다.", "tokens": 0}

    system = (
        "너는 주식 자동매매 시뮬레이션 분석가다. "
        "시뮬레이션 이벤트 로그를 분석하여 다음 4개 섹션으로 설명해:\n"
        "1. **매매 흐름 요약** — 언제 매수/매도했고, 왜 그 판단을 했는지 시간순 서술\n"
        "2. **주문 실행 분석** — 지정가 체결 속도, 부분체결 발생 여부, TTL 만료 처리, "
        "잔량 취소/시장가 재주문 과정\n"
        "3. **리스크 관리** — 손절/트레일링 발동 여부, MDD, 적정성 평가\n"
        "4. **개선 제안** — 전략 파라미터나 주문 실행 설정 조정 방향\n"
        "간결하게, 핵심만. 한국어로."
    )

    # 이벤트 로그 (최근 150건으로 제한 — 토큰 관리)
    recent = events[-150:]
    lines = [f"[bar {e.get('bar', '?')}] {e.get('type', '?')} {e.get('symbol', '')} {e.get('detail', '')}"
             for e in recent]
    prompt = (
        f"시뮬레이션 이벤트 {len(events)}건 (최근 {len(recent)}건 표시):\n"
        + "\n".join(lines)
        + f"\n\n최종 상태: PnL={final_state.get('pnl', 0):+,.0f}원 "
        f"({final_state.get('pnl_pct', 0):+.2f}%), "
        f"현금={final_state.get('cash', 0):,.0f}원, "
        f"포지션={len(final_state.get('positions', {}))}종목"
    )

    try:
        result = await chat_simple(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=2000,
        )
        return {
            "analysis": result.text,
            "tokens": result.input_tokens + result.output_tokens,
        }
    except Exception as e:
        logger.error("시뮬 분석 LLM 호출 실패: %s", e)
        return {"analysis": f"분석 실패: {e}", "tokens": 0}


# ── 히스토리 CRUD ──────────────────────────────────────────


@router.post("/runs")
async def save_run(body: dict):
    """시뮬레이션 결과 저장."""
    import uuid as _uuid
    from sqlalchemy import text
    from app.core.database import async_session

    run_id = str(_uuid.uuid4())
    async with async_session() as db:
        await db.execute(text(
            "INSERT INTO sim_replay_runs "
            "(id, symbol, interval, strategy_preset, factor_id, data_source, "
            " config, total_bars, total_events, pnl, pnl_pct, events, final_state, analysis) "
            "VALUES (:id, :symbol, :interval, :preset, :factor_id, :source, "
            " CAST(:config AS jsonb), :bars, :n_events, :pnl, :pnl_pct, CAST(:events AS jsonb), CAST(:state AS jsonb), :analysis)"
        ), {
            "id": run_id,
            "symbol": body.get("symbol", ""),
            "interval": body.get("interval", "5m"),
            "preset": body.get("strategy_preset", ""),
            "factor_id": body.get("factor_id") or None,
            "source": body.get("data_source", "synthetic"),
            "config": json.dumps(body.get("config", {})),
            "bars": body.get("total_bars", 0),
            "n_events": body.get("total_events", 0),
            "pnl": body.get("pnl"),
            "pnl_pct": body.get("pnl_pct"),
            "events": json.dumps(body.get("events", [])),
            "state": json.dumps(body.get("final_state", {})),
            "analysis": body.get("analysis", ""),
        })
        await db.commit()

    return {"id": run_id, "success": True}


@router.get("/runs")
async def list_runs(limit: int = 30, offset: int = 0):
    """히스토리 목록 (최신순, events 제외 — 가볍게)."""
    from sqlalchemy import text
    from app.core.database import async_session

    async with async_session() as db:
        result = await db.execute(text(
            "SELECT id, symbol, interval, strategy_preset, factor_id, data_source, "
            "       total_bars, total_events, pnl, pnl_pct, analysis IS NOT NULL as has_analysis, "
            "       created_at "
            "FROM sim_replay_runs "
            "ORDER BY created_at DESC "
            "LIMIT :limit OFFSET :offset"
        ), {"limit": limit, "offset": offset})
        rows = result.fetchall()

        count_r = await db.execute(text("SELECT COUNT(*) FROM sim_replay_runs"))
        total = count_r.scalar() or 0

    return {
        "items": [
            {
                "id": str(r[0]),
                "symbol": r[1],
                "interval": r[2],
                "strategy_preset": r[3],
                "factor_id": str(r[4]) if r[4] else None,
                "data_source": r[5],
                "total_bars": r[6],
                "total_events": r[7],
                "pnl": float(r[8]) if r[8] is not None else None,
                "pnl_pct": r[9],
                "has_analysis": r[10],
                "created_at": str(r[11]),
            }
            for r in rows
        ],
        "total": total,
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """히스토리 상세 (events + analysis 포함)."""
    from sqlalchemy import text
    from app.core.database import async_session

    async with async_session() as db:
        result = await db.execute(text(
            "SELECT id, symbol, interval, strategy_preset, factor_id, data_source, "
            "       config, total_bars, total_events, pnl, pnl_pct, "
            "       events, final_state, analysis, created_at "
            "FROM sim_replay_runs WHERE id = :id"
        ), {"id": run_id})
        r = result.fetchone()

    if not r:
        from fastapi import HTTPException
        raise HTTPException(404, "시뮬레이션 기록 없음")

    return {
        "id": str(r[0]),
        "symbol": r[1],
        "interval": r[2],
        "strategy_preset": r[3],
        "factor_id": str(r[4]) if r[4] else None,
        "data_source": r[5],
        "config": r[6],
        "total_bars": r[7],
        "total_events": r[8],
        "pnl": float(r[9]) if r[9] is not None else None,
        "pnl_pct": r[10],
        "events": r[11],
        "final_state": r[12],
        "analysis": r[13],
        "created_at": str(r[14]),
    }


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str):
    """히스토리 삭제."""
    from sqlalchemy import text
    from app.core.database import async_session

    async with async_session() as db:
        await db.execute(text("DELETE FROM sim_replay_runs WHERE id = :id"), {"id": run_id})
        await db.commit()
    return {"success": True}
