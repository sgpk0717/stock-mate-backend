"""매매 로그 Markdown 파일 저장.

market_close 시 호출되어 trade_log + decision_log를
logs/{날짜}_{세션ID}.md 파일로 영속화한다.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 로그 디렉토리: 프로젝트 루트/logs/trading/
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "trading"


def save_session_log(
    session_id: str,
    mode: str,
    strategy_name: str,
    initial_capital: float,
    position_size_pct: float,
    max_positions: int,
    trade_log: list[dict],
    decision_log: list[dict],
    cost_config: Any = None,
) -> str | None:
    """세션의 매매 기록을 Markdown 파일로 저장. 저장된 경로를 반환."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        short_id = session_id[:8]
        filename = f"{today}_{short_id}.md"
        filepath = LOG_DIR / filename

        buys = [t for t in trade_log if t.get("side") == "BUY"]
        sells = [t for t in trade_log if t.get("side") == "SELL"]

        # 성과 계산
        pnls = [t.get("pnl_pct") or 0 for t in sells]
        pnl_amounts = [t.get("pnl_amount") or 0 for t in sells]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_pnl = sum(pnl_amounts)
        holds = [t["holding_minutes"] for t in sells if (t.get("holding_minutes") or 0) > 0]

        # 수수료 정보
        buy_comm = getattr(cost_config, "buy_commission", 0.00015) if cost_config else 0.00015
        sell_comm = getattr(cost_config, "sell_commission", 0.00215) if cost_config else 0.00215
        slippage = getattr(cost_config, "slippage_pct", 0.001) if cost_config else 0.001

        lines: list[str] = []
        w = lines.append

        # 헤더
        w(f"# Paper Trading Report — {today}")
        w("")
        w(f"- **세션**: `{session_id}`")
        w(f"- **모드**: {mode}")
        w(f"- **전략**: {strategy_name}")
        w(f"- **초기자본**: {initial_capital:,.0f}원")
        w(f"- **종목당 배분**: {position_size_pct * 100:.0f}% ({initial_capital * position_size_pct:,.0f}원)")
        w(f"- **최대 포지션**: {max_positions}")
        w(f"- **수수료**: 매수 {buy_comm * 100:.3f}% + 매도 {sell_comm * 100:.3f}% | 슬리피지 {slippage * 100:.1f}%")
        w("")

        # 성과 요약
        w("## 성과 요약")
        w("")
        w(f"| 항목 | 값 |")
        w(f"|------|-----|")
        w(f"| 총 거래 | {len(trade_log)}건 (매수 {len(buys)} / 매도 {len(sells)}) |")
        win_rate = (len(wins) / len(sells) * 100) if sells else 0
        w(f"| 승률 | {win_rate:.1f}% ({len(wins)}승 / {len(losses)}패) |")
        w(f"| 총 PnL | **{total_pnl:+,.0f}원** ({total_pnl / initial_capital * 100:+.4f}%) |")
        if wins:
            w(f"| 평균 수익 | +{sum(wins) / len(wins):.4f}% |")
        if losses:
            w(f"| 평균 손실 | {sum(losses) / len(losses):.4f}% |")
        if holds:
            w(f"| 보유시간 | {min(holds):.0f}분 ~ {max(holds):.0f}분 (평균 {sum(holds) / len(holds):.1f}분) |")
        w("")

        # 매수 내역
        w("## 매수 내역")
        w("")
        w("| # | 봉 시각 | 종목 | 코드 | 수량 | 매수가 | 투입금 | alpha | RSI | 사유 |")
        w("|---|--------|------|------|------|--------|--------|-------|-----|------|")
        for i, t in enumerate(buys, 1):
            snap = t.get("snapshot") or {}
            sz = t.get("sizing") or {}
            conds = (t.get("conditions") or {}).get("buy_conditions") or []
            alpha_key = next((k for k in snap if k.startswith("alpha_")), None)
            alpha_v = f"{snap[alpha_key]:.4f}" if alpha_key and snap.get(alpha_key) is not None else "-"
            rsi_v = f"{snap['rsi']:.1f}" if snap.get("rsi") is not None else "-"
            reason = ", ".join(f"`{c['indicator']}{c['op']}{c['threshold']}` (={c['actual']})" for c in conds)
            ts = (t.get("timestamp") or "")[:19]
            w(f"| {i} | {ts} | {t.get('name', '')} | {t.get('symbol', '')} | {t.get('qty', 0)} | {t.get('price', 0):,.0f} | {sz.get('total_cost', 0):,.0f} | {alpha_v} | {rsi_v} | {reason} |")
        w("")

        # 매도 내역
        w("## 매도 내역")
        w("")
        w("| # | 봉 시각 | 종목 | 코드 | 수량 | 매도가 | PnL% | PnL금액 | 보유 | 진입봉 | 사유 |")
        w("|---|--------|------|------|------|--------|------|---------|------|--------|------|")
        for i, t in enumerate(sells, 1):
            pc = t.get("position_context") or {}
            entry = (pc.get("entry_candle_dt") or pc.get("entry_date") or "-")[:19]
            pnl_pct = t.get("pnl_pct")
            pct_str = f"{pnl_pct:+.4f}%" if pnl_pct is not None else "-"
            amt = t.get("pnl_amount")
            amt_str = f"{amt:+,.0f}" if amt is not None else "-"
            hold = t.get("holding_minutes")
            hold_str = f"{hold:.0f}분" if hold and hold > 0 else "0분"
            reason = (t.get("reason") or "")[:50]
            ts = (t.get("timestamp") or "")[:19]
            w(f"| {i} | {ts} | {t.get('name', '')} | {t.get('symbol', '')} | {t.get('qty', 0)} | {t.get('price', 0):,.0f} | {pct_str} | {amt_str} | {hold_str} | {entry} | {reason} |")
        w("")

        # 판단 로그 요약
        if decision_log:
            action_counts: dict[str, int] = {}
            for d in decision_log:
                a = d.get("action", "UNKNOWN")
                action_counts[a] = action_counts.get(a, 0) + 1

            w("## 판단 로그 요약")
            w("")
            w(f"총 {len(decision_log)}건")
            w("")
            w("| 액션 | 건수 |")
            w("|------|------|")
            for action, cnt in sorted(action_counts.items()):
                w(f"| {action} | {cnt} |")
            w("")

        # 판단 로그 상세 (스킵 제외, 실행된 매매만)
        executed = [d for d in decision_log if d.get("action") in ("BUY", "SELL", "RISK_STOP", "RISK_TRAIL")]
        if executed:
            w("## 판단 상세 (실행된 매매)")
            w("")
            for d in executed:
                ts = (d.get("timestamp") or "")[:19]
                name = d.get("name") or d.get("symbol", "")
                w(f"### {d.get('action')} {name} ({d.get('symbol', '')}) — {ts}")
                w("")
                w(f"- **사유**: {d.get('reason', '')}")

                # 조건 상세
                conds = d.get("conditions")
                if conds:
                    for side_key in ("buy_conditions", "sell_conditions"):
                        side_conds = conds.get(side_key, [])
                        if side_conds:
                            w(f"- **{side_key}**:")
                            for c in side_conds:
                                status = "충족" if c.get("met") else "미충족"
                                w(f"  - `{c['indicator']}` {c['op']} {c['threshold']} → 실제: {c['actual']} ({status})")

                # 스냅샷
                snap = d.get("snapshot")
                if snap:
                    important = {k: v for k, v in snap.items() if k in (
                        "close", "rsi", "macd_hist", "volume_ratio", "atr_14",
                    ) or k.startswith("alpha_")}
                    if important:
                        parts = [f"{k}={v}" for k, v in important.items()]
                        w(f"- **지표**: {', '.join(parts)}")

                # 사이징
                sz = d.get("sizing")
                if sz and d.get("action") == "BUY":
                    w(f"- **사이징**: 배분 {sz.get('alloc_cash_limited', 0):,.0f}원 / 매수가 {sz.get('buy_price_effective', 0):,.0f} = {sz.get('qty', 0)}주 (투입 {sz.get('total_cost', 0):,.0f}원)")

                # 리스크
                risk = d.get("risk")
                if risk:
                    w(f"- **리스크**: {risk}")
                w("")

        # 스킵 판단 예시 (최대 10건)
        skips = [d for d in decision_log if d.get("action", "").startswith("SKIP")]
        if skips:
            w("## 스킵 판단 (상위 10건)")
            w("")
            w("| 시각 | 종목 | 액션 | 사유 |")
            w("|------|------|------|------|")
            for d in skips[:10]:
                ts = (d.get("timestamp") or "")[:19]
                name = d.get("name") or d.get("symbol", "")
                w(f"| {ts} | {name} | {d.get('action', '')} | {d.get('reason', '')[:60]} |")
            if len(skips) > 10:
                w(f"| ... | ... | ... | 외 {len(skips) - 10}건 |")
            w("")

        content = "\n".join(lines)
        filepath.write_text(content, encoding="utf-8")
        logger.info("매매 로그 저장: %s (%d건)", filepath, len(trade_log))
        return str(filepath)

    except Exception as e:
        logger.error("매매 로그 저장 실패: %s", e, exc_info=True)
        return None
