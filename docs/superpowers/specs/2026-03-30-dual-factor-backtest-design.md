# 듀얼팩터 백테스트 설계

## Context

일봉 팩터로 매수 후보 종목을 선정하고, 5분봉 팩터로 정확한 매수/매도 시점을 결정하는 듀얼 타임프레임 전략.
핵심 제약: 모든 분봉 데이터를 사전 로딩하지 않고, 매매 가능성이 있는 종목의 당일 분봉만 온디맨드 로딩.

## 동작 흐름

```
매일 장 시작 전:
  1. 일봉 팩터로 전체 종목 랭킹 → 상위 N% = 감시 리스트
  2. 감시 리스트 종목만 당일 5분봉 데이터 로딩 (DB에서)
  3. 5분봉 팩터로 매수 시그널 확인 → 시그널 발생 시 매수
  4. 보유 종목도 5분봉 팩터로 매도 시그널 확인 → 시그널 발생 시 매도
  5. 당일 매매 안 이루어지면 → 다음 날로

분봉 데이터 없는 기간 (1년+ 전):
  → 일봉 팩터만으로 기존 로직 (주간 리밸런싱) 적용
```

## 데이터 로딩 전략

**사전 로딩**: 일봉 OHLCV + enrichment (전체 기간, 기존과 동일)
**온디맨드**: 5분봉은 매매 후보 종목 × 당일만 (날짜별 lazy loading)

```python
# 의사 코드
for trading_date in all_dates:
    # Phase 1: 일봉 팩터 랭킹 (사전 로딩된 데이터 사용)
    daily_rankings = compute_daily_factor_rank(daily_df, trading_date)
    watchlist = daily_rankings[:top_n_symbols]  # 상위 N종목

    # Phase 2: 분봉 데이터 존재 여부 확인
    if has_intraday_data(trading_date):
        # 감시 종목 + 보유 종목의 당일 5분봉만 로딩
        targets = set(watchlist) | set(current_positions.keys())
        intraday_df = load_intraday_candles(targets, trading_date, "5m")

        # 5분봉 팩터로 매수/매도 시점 결정
        for bar in intraday_bars:
            intraday_signals = compute_intraday_factor(intraday_df, bar)
            execute_trades(intraday_signals, ...)
    else:
        # 분봉 없음 → 일봉 리밸런싱 (기존 로직)
        if is_rebalance_day(trading_date):
            rebalance_portfolio(daily_rankings, ...)
```

## 파라미터

```python
async def run_factor_backtest(
    expression_str: str,          # 일봉 팩터 수식
    intraday_expression_str: str | None = None,  # 5분봉 팩터 수식 (없으면 일봉 전용)
    intraday_interval: str = "5m",
    intraday_entry_threshold: float = 0.8,  # 5분봉 팩터 랭크 상위 80% 이상이면 매수
    intraday_exit_threshold: float = 0.2,   # 5분봉 팩터 랭크 하위 20% 이면 매도
    ...
)
```

## 분봉 데이터 로딩 최적화

1. **날짜별 배치**: 한 날짜의 감시 종목 전부를 한번에 쿼리
   ```sql
   SELECT * FROM stock_candles
   WHERE interval='5m' AND symbol IN (:symbols) AND dt::date = :date
   ORDER BY symbol, dt
   ```

2. **캐시**: 같은 날짜를 다시 쿼리하지 않도록 LRU 캐시 (최근 5일분)

3. **데이터 존재 체크**:
   ```sql
   SELECT MIN(dt::date) FROM stock_candles WHERE interval='5m' LIMIT 1
   ```
   이 날짜 이전은 일봉 전용 모드로 자동 전환.

## 구현 순서

1. `factor_backtest.py`에 `run_dual_factor_backtest()` 함수 추가
2. 5분봉 온디맨드 로더 함수 추가 (`_load_intraday_for_date()`)
3. 프론트: FactorBacktestConfig에 "듀얼팩터" 모드 토글 + 5분봉 팩터 선택
4. API: `AlphaFactorBacktestRequest`에 `intraday_factor_id` 필드 추가
5. 라우터: 듀얼팩터 모드일 때 `run_dual_factor_backtest()` 호출

## 미구현 사항 (향후)

- 5분봉 팩터 자체의 ensure_alpha_features (분봉 전용 피처)
- 분봉 데이터가 부분적으로만 있는 종목 처리
- 장중 시간대별 가중치 (오전/오후 차이)
