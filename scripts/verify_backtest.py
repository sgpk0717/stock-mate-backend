"""백테스트 트레이드 교차 검증 스크립트.

핵심: entry_date = 체결일(T+1), entry_snapshot = 시그널일(T) 데이터.
따라서 시그널일 = entry_date 직전 거래일.

검증 항목:
1. entry_snapshot.close == DB 시그널일 종가
2. exit_snapshot.close == DB exit 시그널일 종가
3. entry/exit RSI가 전체 히스토리 기반 Wilder RSI와 ±0.5 이내
4. entry_date, exit_date가 평일(거래일)
"""

import asyncio
import sys
from datetime import date, datetime, timedelta, timezone

import asyncpg

_KST = timezone(timedelta(hours=9))

DSN = "postgresql://stockmate:stockmate@postgres:5432/stockmate"

# 검증 대상 10건
SAMPLES = [
    {"symbol": "000040", "name": "KR모터스", "entry_date": "2024-01-03", "exit_date": "2024-02-22",
     "entry_close": 1067, "entry_rsi": 29.06, "exit_close": 1433, "exit_rsi": 76.28},
    {"symbol": "001750", "name": "한양증권", "entry_date": "2024-01-03", "exit_date": "2024-02-02",
     "entry_close": 9150, "entry_rsi": 26.56, "exit_close": 9740, "exit_rsi": 77.01},
    {"symbol": "000230", "name": "일동홀딩스", "entry_date": "2024-02-02", "exit_date": "2024-08-21",
     "entry_close": 9260, "entry_rsi": 28.01, "exit_close": 11360, "exit_rsi": 80.40},
    {"symbol": "002960", "name": "한국쉘석유", "entry_date": "2024-01-03", "exit_date": "2024-02-23",
     "entry_close": 226000, "entry_rsi": 24.17, "exit_close": 241000, "exit_rsi": 78.03},
    {"symbol": "000545", "name": "흥국화재우", "entry_date": "2024-06-20", "exit_date": "2024-12-26",
     "entry_close": 6000, "entry_rsi": 29.04, "exit_close": 6000, "exit_rsi": 78.93},
    {"symbol": "000670", "name": "영풍", "entry_date": "2024-06-18", "exit_date": "2024-09-19",
     "entry_close": 32697, "entry_rsi": 26.40, "exit_close": 36372, "exit_rsi": 77.17},
    {"symbol": "000850", "name": "화천기공", "entry_date": "2024-01-03", "exit_date": "2024-02-20",
     "entry_close": 31300, "entry_rsi": 29.08, "exit_close": 33250, "exit_rsi": 70.94},
    {"symbol": "000300", "name": "DH오토넥스", "entry_date": "2024-01-03", "exit_date": "2024-02-19",
     "entry_close": 2891, "entry_rsi": 27.86, "exit_close": 4494, "exit_rsi": 70.18},
    {"symbol": "001800", "name": "오리온홀딩스", "entry_date": "2024-01-03", "exit_date": "2024-06-18",
     "entry_close": 14420, "entry_rsi": 29.04, "exit_close": 15900, "exit_rsi": 70.71},
    {"symbol": "002995", "name": "금호건설우", "entry_date": "2024-02-19", "exit_date": "2024-06-20",
     "entry_close": 13500, "entry_rsi": 23.32, "exit_close": 15610, "exit_rsi": 77.46},
]


def _to_kst_date(dt_val: datetime) -> date:
    if hasattr(dt_val, "astimezone"):
        return dt_val.astimezone(_KST).date()
    if hasattr(dt_val, "date"):
        return dt_val.date()
    return dt_val


def compute_wilder_rsi(closes: list[float], period: int = 14) -> float | None:
    """전체 종가 리스트의 마지막 봉 Wilder RSI 계산."""
    if len(closes) < period + 1:
        return None

    alpha = 1.0 / period
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = alpha * gain + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def find_signal_date(sorted_dates: list[date], exec_date: date) -> int | None:
    """체결일(exec_date) 직전 거래일의 인덱스를 반환."""
    for i in range(len(sorted_dates) - 1, -1, -1):
        if sorted_dates[i] < exec_date:
            return i
    return None


async def main():
    conn = await asyncpg.connect(DSN)

    print("=" * 100)
    print("백테스트 트레이드 교차 검증 (10건)")
    print("entry_date = 체결일(T+1), snapshot = 시그널일(T) 데이터")
    print("=" * 100)
    print()

    total_checks = 0
    passed_checks = 0
    failed_details = []

    for idx, sample in enumerate(SAMPLES, 1):
        symbol = sample["symbol"]
        name = sample["name"]
        entry_date = date.fromisoformat(sample["entry_date"])
        exit_date = date.fromisoformat(sample["exit_date"])

        print(f"[{idx}/10] {symbol} {name}")
        print(f"  체결일: Entry={entry_date}  Exit={exit_date}")

        # ── 1. 날짜가 평일인지 확인 ──
        for label, d in [("entry", entry_date), ("exit", exit_date)]:
            total_checks += 1
            if d.weekday() < 5:
                passed_checks += 1
            else:
                failed_details.append(f"  [{idx}] {label}_date {d} 은 주말!")

        # ── 2. DB에서 전체 히스토리 로딩 ──
        rows = await conn.fetch("""
            SELECT dt, close::float8
            FROM stock_candles
            WHERE symbol = $1 AND interval = '1d'
            ORDER BY dt
        """, symbol)

        date_close_map = {}
        for row in rows:
            d = _to_kst_date(row["dt"])
            c = row["close"]
            if c > 0:
                date_close_map[d] = c

        sorted_dates = sorted(date_close_map.keys())
        closes = [date_close_map[d] for d in sorted_dates]

        # ── 3. Entry 검증 ──
        entry_sig_idx = find_signal_date(sorted_dates, entry_date)
        if entry_sig_idx is not None:
            sig_date = sorted_dates[entry_sig_idx]
            sig_close = closes[entry_sig_idx]

            # 종가 비교
            total_checks += 1
            if abs(sig_close - sample["entry_close"]) < 1:
                passed_checks += 1
                print(f"  ✓ Entry close: DB={sig_close:.0f} == BT={sample['entry_close']} (시그널일={sig_date})")
            else:
                failed_details.append(
                    f"  [{idx}] Entry close 불일치: DB={sig_close:.0f} vs BT={sample['entry_close']} (시그널일={sig_date})")
                print(f"  ✗ Entry close: DB={sig_close:.0f} vs BT={sample['entry_close']} (시그널일={sig_date})")

            # RSI 비교 (시그널일까지의 전체 히스토리)
            total_checks += 1
            rsi = compute_wilder_rsi(closes[:entry_sig_idx + 1])
            if rsi is not None:
                diff = abs(rsi - sample["entry_rsi"])
                if diff <= 0.5:
                    passed_checks += 1
                    print(f"  ✓ Entry RSI: 계산={rsi:.2f} ≈ BT={sample['entry_rsi']:.2f} (차이={diff:.2f})")
                else:
                    failed_details.append(
                        f"  [{idx}] Entry RSI 차이: 계산={rsi:.2f} vs BT={sample['entry_rsi']:.2f} (차이={diff:.2f})")
                    print(f"  ✗ Entry RSI: 계산={rsi:.2f} vs BT={sample['entry_rsi']:.2f} (차이={diff:.2f})")
            else:
                print(f"  ? Entry RSI: 데이터 부족 (idx={entry_sig_idx})")
        else:
            total_checks += 2
            failed_details.append(f"  [{idx}] Entry 시그널일을 찾을 수 없음 (entry_date={entry_date})")
            print(f"  ✗ Entry 시그널일을 찾을 수 없음")

        # ── 4. Exit 검증 ──
        # exit도 동일: exit_date=체결일(T+1), exit_snapshot=시그널일(T) 데이터
        # 단, 손절/트레일링 exit은 snapshot에 RSI 없이 close만 있음 → 그 경우 해당일 종가
        exit_sig_idx = find_signal_date(sorted_dates, exit_date)
        if exit_sig_idx is not None:
            sig_date = sorted_dates[exit_sig_idx]
            sig_close = closes[exit_sig_idx]

            # 종가 비교
            total_checks += 1
            if abs(sig_close - sample["exit_close"]) < 1:
                passed_checks += 1
                print(f"  ✓ Exit close: DB={sig_close:.0f} == BT={sample['exit_close']} (시그널일={sig_date})")
            else:
                # 손절/트레일링은 당일 종가가 아닌 다른 메커니즘 → exit_date 당일 확인
                exit_same_idx = None
                for i, d in enumerate(sorted_dates):
                    if d == exit_date:
                        exit_same_idx = i
                        break
                if exit_same_idx is not None and abs(closes[exit_same_idx] - sample["exit_close"]) < 1:
                    passed_checks += 1
                    print(f"  ✓ Exit close: DB={closes[exit_same_idx]:.0f} == BT={sample['exit_close']} (체결일 당일={exit_date})")
                else:
                    failed_details.append(
                        f"  [{idx}] Exit close 불일치: DB(T)={sig_close:.0f}, DB(T+1)={closes[exit_same_idx] if exit_same_idx else '?'} vs BT={sample['exit_close']} (시그널={sig_date})")
                    print(f"  ✗ Exit close: DB(T)={sig_close:.0f} vs BT={sample['exit_close']} (시그널일={sig_date})")

            # RSI 비교
            total_checks += 1
            rsi = compute_wilder_rsi(closes[:exit_sig_idx + 1])
            if rsi is not None:
                diff = abs(rsi - sample["exit_rsi"])
                if diff <= 0.5:
                    passed_checks += 1
                    print(f"  ✓ Exit RSI: 계산={rsi:.2f} ≈ BT={sample['exit_rsi']:.2f} (차이={diff:.2f})")
                else:
                    failed_details.append(
                        f"  [{idx}] Exit RSI 차이: 계산={rsi:.2f} vs BT={sample['exit_rsi']:.2f} (차이={diff:.2f})")
                    print(f"  ✗ Exit RSI: 계산={rsi:.2f} vs BT={sample['exit_rsi']:.2f} (차이={diff:.2f})")
            else:
                print(f"  ? Exit RSI: 데이터 부족")
        else:
            total_checks += 2
            failed_details.append(f"  [{idx}] Exit 시그널일을 찾을 수 없음 (exit_date={exit_date})")
            print(f"  ✗ Exit 시그널일을 찾을 수 없음")

        print()

    await conn.close()

    # ── 결과 요약 ──
    print("=" * 100)
    print(f"검증 결과: {passed_checks}/{total_checks} 통과 ({passed_checks/total_checks*100:.1f}%)")
    print("=" * 100)

    if failed_details:
        print("\n실패 항목:")
        for d in failed_details:
            print(d)
    else:
        print("\n모든 검증 통과!")

    return passed_checks == total_checks


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
