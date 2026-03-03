# Phase 3: 자율 알파 팩토리 — 보완 사항

> Phase 3 전문가 검증 후 도출된 개선 사항.
> CRITICAL 이슈는 즉시 수정 완료. 아래는 향후 개선할 HIGH/MEDIUM/LOW 항목.

## 수정 완료 (CRITICAL)

| # | 항목 | 수정 내용 |
|---|------|----------|
| B-C1 | `_cosine_similarity_batch_safe` 무로깅 | 에러 로깅 추가 (`logger.warning`) |
| B-C2 | embedding 실패 시 DB/캐시 불일치 | `embedding=None` 시 debug 로그 추가 |
| B-C3 | scheduler `data.height == 0` 상태 미갱신 | 빈 데이터 시 cycles_completed 증가 + 브로드캐스트 |
| B-C4 | 빈 문자열 날짜 → `ValueError` | `or ""` 방어 + falsy 체크 강화 |
| B-C5 | main.py factory stop 에러 무시 | `except Exception: pass` → `logger.warning` |
| B-C6 | auto-start 시 빈 설정 전달 | config 기본값(interval, iterations, crossover) 명시 |
| B-C7 | parent_ids에 `factor.name` 사용 | `expression_str[:60]` 기반 식별자로 변경 |
| B-C8 | `load_candles()` 전체 DB 로드 | 최근 1년(`timedelta(days=365)`) 기본 제한 |
| B-C9 | correlation 엔드포인트 `list[str]` 직접 수신 | `CorrelationRequest` Pydantic 모델 래핑 |
| B-C10 | `tournament_select` 중복 부모 허용 | 중복 제거 루프 + fallback |
| B-C11 | `_record_experience` commit 경계 | 문서화 확인 (scheduler commit 보장) |
| F-C1 | `fetchCorrelation` raw array 전송 | `{ factor_ids: [...] }` 객체 래핑 |
| F-C2 | AlphaFactoryControl 에러 표시 없음 | 에러 메시지 UI + 입력 검증 (canStart) |
| F-C3 | CompositeFactorBuilder 에러 표시 없음 | `buildComposite.error` UI 표시 |
| F-C4 | `useFactoryStatus` placeholderData 누락 | `keepPreviousData` 추가 |
| F-C5 | 인라인 `= []` CLAUDE.md 위반 | 모듈 레벨 `EMPTY_FACTORS`, `EMPTY_RUNS` 상수 |

---

## HIGH — 향후 구현 권장

### B-H1: portfolio.py — 팩터별 symbols/기간 매핑

현재 `build_composite_factor`와 `compute_correlation_matrix`는 최근 1년 전체 캔들을 로드한다.
팩터가 속한 mining_run의 symbols/기간을 참조하면 불필요한 데이터 로드를 줄일 수 있다.

```python
# mining_run.config에서 symbols/dates 추출
run_configs = [f.mining_run.config for f in factors if f.mining_run]
symbols = set()
for cfg in run_configs:
    symbols.update(cfg.get("symbols", []))
data = await load_candles(symbols=list(symbols), start_date=...)
```

**영향 범위:** `app/alpha/portfolio.py`

### B-H2: 스케줄러 Graceful Shutdown + 상태 영속화

현재 스케줄러 상태는 인메모리. 서버 재시작 시 초기화된다.
DB에 `alpha_factory_state` 테이블을 추가하여 사이클/팩터 카운트를 영속화하면 운영 안정성 향상.

```python
# shutdown 시 상태 저장
await db.execute(
    update(AlphaFactoryState).values(
        cycles_completed=state.cycles_completed,
        factors_total=state.factors_discovered_total,
    )
)
```

### B-H3: 벡터 메모리 TTL + 크기 제한

경험이 수천 개 이상 축적되면 인메모리 numpy 배열이 커진다.
- TTL 기반 오래된 경험 제거 (예: 90일)
- 최대 캐시 크기 제한 (예: 2000개, LRU)

### B-H4: CompositeFactorResponse에 weights 포함

현재 `CompositeResult`에 `weights` dict가 있지만 프론트 응답(`CompositeFactorResponse`)에 포함되지 않음.
각 구성 팩터의 가중치를 프론트에서 표시하면 사용자 이해도 향상.

**영향 범위:** `schemas.py`, `routers/alpha.py`, `types/alpha.ts`

---

## MEDIUM — 품질 개선

### B-M1: check_orthogonality 마이너 루프 통합

`ExperienceVectorMemory.check_orthogonality()` 메서드가 존재하지만 마이너에서 호출하지 않음.
`_generate_hypothesis()` → `_evaluate_and_mutate()` 사이에 직교성 체크를 삽입하면 중복 팩터 생성을 사전 방지.

```python
if self.vector_memory and not self.vector_memory.check_orthogonality(
    expr_str, hypothesis, threshold=settings.ALPHA_FACTORY_ORTHOGONALITY_THRESHOLD
):
    logger.info("Skipping non-orthogonal factor: %s", expr_str[:60])
    continue
```

### F-M1: WebSocket alpha:factory 프론트 구독

백엔드 스케줄러가 `manager.broadcast("alpha:factory", ...)` 하지만 프론트에서 해당 채널을 구독하지 않음.
WebSocket 구독을 추가하면 5초 폴링 대신 실시간 업데이트 가능.

**영향 범위:** `use-alpha.ts` 또는 별도 `useAlphaFactoryStream()` 훅

### F-M2: FactorLineageTree 툴팁

현재 노드 클릭 시 `onSelect()` 호출만 됨. hover 시 수식/IC/generation 툴팁 표시하면 사용성 향상.

```tsx
<title>{`${node.factor.name}\nIC: ${node.factor.ic_mean}\n${node.factor.expression_str}`}</title>
```

### F-M3: FactorCorrelationHeatmap 접근성

Canvas 기반이므로 스크린 리더 접근 불가. 대안:
- `aria-label` 추가
- 선택적으로 HTML table 대체 렌더링 모드

---

## LOW — 장기 개선

### B-L1: evolution.py 단위 테스트 확장

현재 crossover/mutate에 대한 단위 테스트 없음. 다양한 SymPy 표현식 입력에 대한 테스트 추가:
- 단순 수식: `rsi + volume_ratio`
- 중첩 함수: `log(Abs(rsi - 50))`
- 상수 포함: `rsi * 0.5 + volume_ratio * 0.3`
- 깊이 제한 초과 케이스

### B-L2: memory.py 벤치마크

1000/5000/10000 경험에 대한 `retrieve_similar()` 성능 벤치마크.
현재 인메모리 코사인 유사도는 O(N)이므로 대규모에서 성능 확인 필요.

### F-L1: 팩터 이름 자동 생성 개선

현재 팩터 이름이 `alpha_0_0`, `alpha_crossover_1` 등 기계적. Claude에게 의미 있는 이름 생성 요청:
- "모멘텀 반전 RSI"
- "거래량 브레이크아웃"
