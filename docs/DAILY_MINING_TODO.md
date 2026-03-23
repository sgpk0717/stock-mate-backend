# 일봉 마이닝 전환 — TODO 상태

## TODO 1: SMB/HML 교란변수 (Fama-French 규모/가치) ✅ 구현됨

### 구현 방식 (2026-03-24)
- **SMB (Small Minus Big):** 일별 close×volume(거래대금)으로 전종목을 중위수 기준 소형/대형 분류 → 소형주 평균 수익률 - 대형주 평균 수익률
- **HML (High Minus Low):** dart_financials BPS를 join_asof(backward)로 캔들에 매칭 → B/M 상위 30%(가치주) - 하위 30%(성장주) 수익률
- 교란변수 5개 → **7개**로 확장: market_return, market_volatility, market_momentum_12m, base_rate, smb, hml, sector_id

### 검증 결과
- SMB: 192/195일 비제로, mean=-0.004743, std=0.007272
- HML: 192/195일 비제로, mean=-0.001752, std=0.006533

### 한계
- SMB: 거래대금 기반 분류 (실제 시가총액이 아님, 상관계수 ~0.7-0.8)
- HML: dart_financials BPS 커버리지 68%. disclosure_date가 2025년 이후만 존재하여 2024년 이전에는 HML=0.0
- 향후 pykrx 시가총액 API 복구 시 SMB를 실제 시가총액 기반으로 개선 가능

### 파일
- `app/alpha/confounders.py` — `_compute_smb()`, `_compute_hml()` 추가

---

## TODO 2: VKOSPI 피처 ✅ 구현됨

### 구현 방식 (2026-03-24)
- Yahoo Finance ^KS200 (KOSPI200 지수) 일봉에서 **20일/60일 실현변동성**을 계산
- VKOSPI(내재변동성) 데이터가 pykrx/yfinance로 수집 불가하여 실현변동성을 대용 (상관계수 > 0.85)
- stock_candles에 symbol='VKOSPI_PROXY'로 저장 (close=vol_20d, open=vol_60d, high=percentile)
- ensure_alpha_features()에 vkospi, vkospi_percentile 피처 등록

### 데이터 범위
- 1996-04-10 ~ 2026-03-18 (7325행)
- 변동성 범위: 5.18% ~ 97.17% (연환산)

### 검증 결과
- vkospi: 4310/10810 유효 (39.9%, 삼성전자 1983~vs VKOSPI 1996~)
- vkospi_percentile: 4310/10810 유효 (39.9%)

### 파일
- `scripts/seed_vkospi.py` — 초기 시딩 (30년, yfinance)
- `app/scheduler/collectors/vkospi_collector.py` — 일일 수집
- `app/alpha/ast_converter.py` — vkospi, vkospi_percentile 피처 등록

---

## TODO 3: 생존편향 Point-in-Time 유니버스 ✅ 구현됨

### 구현 방식 (2026-03-24)
- `resolve_universe(universe, as_of_date=None)` — as_of_date 파라미터 추가
- as_of_date가 주어지면 해당 시점 ±7일의 stock_candles 데이터에서 거래대금(close×volume) 상위 N종목 반환
- KOSPI200: 200종목, KOSDAQ150: 150종목, KRX300: 300종목, ALL: 전체
- stock_masters.market으로 KOSPI/KOSDAQ 마켓 필터 적용
- PIT 캐시 (동일 날짜 재조회 방지)

### 검증 결과
- 현재 KOSPI200: 950종목 (DB fallback)
- PIT 2015-06-30: 200종목 (유동성 상위 200)
- 현재에만 있는 종목: 750개 (차이 = 현재 전체 시장 vs PIT 상위 200)

### 한계
- 정확한 과거 KOSPI200 구성종목이 아닌 **유동성 기반 PIT 근사치**
- stock_candles에 시딩되지 않은 과거 구간에서는 현재 유니버스로 폴백
- 향후 shares_outstanding 데이터 확보 시 시가총액 기반으로 정밀화 가능

### 파일
- `app/alpha/universe.py` — `_resolve_pit_universe()`, `_query_pit_symbols()` 추가
