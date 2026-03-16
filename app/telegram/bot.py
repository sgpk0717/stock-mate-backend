"""Telegram Bot — 명령어 핸들러 + 인라인 키보드 승인.

httpx 기반 경량 구현 (python-telegram-bot 의존성 없음).
설계서 §11.4의 6개 명령어 + L3/L4 승인 메커니즘.

명령어:
  /status  — 현재 워크플로우 상태
  /stop    — 긴급 정지
  /resume  — 정지 해제
  /skip    — 오늘 매매 스킵
  /report  — 오늘 매매 리포트
  /factors — 최고 팩터 목록
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import date

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.telegram.org/bot{token}"
_poll_task: asyncio.Task | None = None
_last_update_id: int = 0

# 승인 대기 큐: callback_data → {description, resolver_future}
_approval_queue: dict[str, dict] = {}


async def _api(method: str, **kwargs) -> dict | None:
    """Telegram Bot API 호출."""
    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        return None
    url = f"{_BASE_URL.format(token=token)}/{method}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=kwargs)
            data = resp.json()
            if not data.get("ok"):
                logger.warning("Telegram API %s failed: %s", method, data)
                return None
            return data.get("result")
    except Exception as e:
        logger.error("Telegram API %s error: %s", method, e)
        return None


# ── 발송 로깅 ──


async def _log_message(
    category: str,
    caller: str,
    text: str,
    chat_id: str,
    status: str,
    error_message: str | None = None,
    telegram_message_id: int | None = None,
) -> None:
    """DB에 텔레그램 발송 내역 기록. 실패해도 발송에 영향 없음."""
    try:
        from app.core.database import async_session
        from app.telegram.models import TelegramMessageLog

        async with async_session() as db:
            log = TelegramMessageLog(
                category=category,
                caller=caller,
                text=text[:4000],
                chat_id=chat_id,
                status=status,
                error_message=error_message,
                telegram_message_id=telegram_message_id,
            )
            db.add(log)
            await db.commit()
    except Exception as e:
        logger.debug("Telegram log DB write failed: %s", e)


async def send_message(
    text: str,
    chat_id: str = "",
    reply_markup: dict | None = None,
    *,
    category: str = "system",
    caller: str = "",
) -> dict | None:
    """메시지 전송 + DB 로깅."""
    cid = chat_id or settings.TELEGRAM_CHAT_ID
    if not cid:
        await _log_message(category, caller, text, "", "skipped", "chat_id 없음")
        return None
    kwargs = {"chat_id": cid, "text": text, "parse_mode": "HTML"}
    if reply_markup:
        kwargs["reply_markup"] = reply_markup
    result = await _api("sendMessage", **kwargs)
    if result:
        await _log_message(
            category, caller, text, cid, "success",
            telegram_message_id=result.get("message_id"),
        )
    else:
        await _log_message(category, caller, text, cid, "failed", "API 호출 실패")
    return result


async def request_approval(
    description: str,
    timeout_seconds: int = 300,
) -> bool:
    """인라인 키보드로 사용자 승인 요청 (L3/L4).

    Returns True if approved, False if rejected or timeout.
    """
    task_id = str(uuid.uuid4())[:8]
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    _approval_queue[f"approve:{task_id}"] = {"future": future, "desc": description}
    _approval_queue[f"reject:{task_id}"] = {"future": future, "desc": description}

    markup = {
        "inline_keyboard": [[
            {"text": "승인", "callback_data": f"approve:{task_id}"},
            {"text": "거부", "callback_data": f"reject:{task_id}"},
        ]]
    }

    await send_message(
        f"<b>[승인 필요]</b>\n{description}",
        reply_markup=markup,
        category="approval",
        caller="bot.request_approval",
    )

    try:
        result = await asyncio.wait_for(future, timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        logger.info("Approval timeout for task %s", task_id)
        await send_message(
            f"[타임아웃] 승인 요청 만료: {description[:50]}",
            category="approval",
            caller="bot.request_approval",
        )
        return False
    finally:
        _approval_queue.pop(f"approve:{task_id}", None)
        _approval_queue.pop(f"reject:{task_id}", None)


# ── 명령어 핸들러 ──


async def _handle_status(chat_id: str) -> None:
    from app.workflow.orchestrator import get_orchestrator
    orch = get_orchestrator()
    status = await orch.get_status()

    phase = status.get("phase", "UNKNOWN")
    st = status.get("status", "")
    factor_id = status.get("selected_factor_id", "없음")
    trades = status.get("trade_count", 0)
    pnl = status.get("pnl_pct")
    mining = status.get("mining_running", False)

    text = (
        f"<b>워크플로우 상태</b>\n"
        f"Phase: <code>{phase}</code>\n"
        f"Status: {st}\n"
        f"팩터: {factor_id[:8] if factor_id and factor_id != '없음' else '없음'}\n"
        f"거래: {trades}건\n"
        f"PnL: {pnl:+.2f}%\n" if pnl is not None else
        f"<b>워크플로우 상태</b>\n"
        f"Phase: <code>{phase}</code>\n"
        f"Status: {st}\n"
        f"팩터: {factor_id[:8] if factor_id and factor_id != '없음' else '없음'}\n"
        f"거래: {trades}건\n"
        f"PnL: -\n"
    )
    text += f"마이닝: {'실행중' if mining else '중지'}"
    await send_message(text, chat_id, category="command_response", caller="bot._handle_status")


async def _handle_stop(chat_id: str) -> None:
    from app.workflow.orchestrator import get_orchestrator
    orch = get_orchestrator()
    result = await orch.handle_emergency_stop()
    if result.get("success"):
        await send_message(
            f"긴급 정지 완료. 이전 상태: {result.get('previous')}\n전량 청산 실행됨.",
            chat_id,
            category="command_response",
            caller="bot._handle_stop",
        )
    else:
        await send_message(
            f"정지 실패: {result.get('message')}",
            chat_id,
            category="command_response",
            caller="bot._handle_stop",
        )


async def _handle_resume(chat_id: str) -> None:
    from app.workflow.orchestrator import get_orchestrator
    orch = get_orchestrator()
    result = await orch.handle_resume()
    if result.get("success"):
        await send_message(
            "정지 해제 → IDLE 복귀.",
            chat_id,
            category="command_response",
            caller="bot._handle_resume",
        )
    else:
        await send_message(
            f"실패: {result.get('message')}",
            chat_id,
            category="command_response",
            caller="bot._handle_resume",
        )


async def _handle_skip(chat_id: str) -> None:
    from app.workflow.orchestrator import get_orchestrator
    orch = get_orchestrator()
    result = await orch.handle_reset()
    await send_message(
        f"오늘 매매 스킵. ({result.get('previous', '?')} → IDLE)",
        chat_id,
        category="command_response",
        caller="bot._handle_skip",
    )


async def _handle_report(chat_id: str) -> None:
    from sqlalchemy import select

    from app.core.database import async_session
    from app.workflow.models import WorkflowRun

    async with async_session() as session:
        stmt = select(WorkflowRun).where(WorkflowRun.date == date.today())
        result = await session.execute(stmt)
        run = result.scalar_one_or_none()

    if not run:
        await send_message(
            "오늘 워크플로우 기록 없음.",
            chat_id,
            category="command_response",
            caller="bot._handle_report",
        )
        return

    review = run.review_summary or {}
    pnl = run.pnl_pct
    trades = run.trade_count or 0
    win_rate = review.get("win_rate", "-")

    text = (
        f"<b>일일 리포트 ({date.today()})</b>\n"
        f"Phase: {run.phase} / Status: {run.status}\n"
        f"거래: {trades}건\n"
        f"PnL: {pnl:+.2f}%\n" if pnl is not None else
        f"<b>일일 리포트 ({date.today()})</b>\n"
        f"Phase: {run.phase} / Status: {run.status}\n"
        f"거래: {trades}건\n"
        f"PnL: -\n"
    )
    text += f"승률: {win_rate}%"
    await send_message(text, chat_id, category="command_response", caller="bot._handle_report")


async def _handle_factors(chat_id: str) -> None:
    from app.core.database import async_session
    from app.workflow.auto_selector import select_best_factors

    async with async_session() as session:
        factors = await select_best_factors(session, limit=5)

    if not factors:
        await send_message(
            "매매 가능 팩터 없음.",
            chat_id,
            category="command_response",
            caller="bot._handle_factors",
        )
        return

    lines = ["<b>Top 팩터</b>"]
    for i, f in enumerate(factors, 1):
        factor = f["factor"]
        lines.append(
            f"{i}. {factor.name}\n"
            f"   IC={factor.ic_mean:.4f} Sharpe={factor.sharpe:.2f} "
            f"Score={f['score']:.4f}"
        )
    await send_message(
        "\n".join(lines),
        chat_id,
        category="command_response",
        caller="bot._handle_factors",
    )


_COMMANDS = {
    "/status": _handle_status,
    "/stop": _handle_stop,
    "/resume": _handle_resume,
    "/skip": _handle_skip,
    "/report": _handle_report,
    "/factors": _handle_factors,
}


# ── 폴링 루프 ──


async def _process_update(update: dict) -> None:
    """단일 Telegram update 처리."""
    # 콜백 쿼리 (인라인 키보드)
    callback = update.get("callback_query")
    if callback:
        data = callback.get("data", "")
        cb_id = callback.get("id")

        entry = _approval_queue.get(data)
        if entry and not entry["future"].done():
            approved = data.startswith("approve:")
            entry["future"].set_result(approved)
            answer_text = "승인됨" if approved else "거부됨"
            await _api("answerCallbackQuery", callback_query_id=cb_id, text=answer_text)
            await send_message(
                f"[{answer_text}] {entry['desc'][:50]}",
                category="approval",
                caller="bot._process_update",
            )
        else:
            await _api("answerCallbackQuery", callback_query_id=cb_id, text="만료됨")
        return

    # 텍스트 명령어
    message = update.get("message", {})
    text = (message.get("text") or "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))

    if not text or not chat_id:
        return

    # 허용된 chat_id만 처리
    allowed = settings.TELEGRAM_CHAT_ID
    if allowed and chat_id != allowed:
        logger.warning("Unauthorized Telegram chat: %s", chat_id)
        return

    cmd = text.split()[0].split("@")[0].lower()  # /status@botname → /status
    handler = _COMMANDS.get(cmd)
    if handler:
        try:
            await handler(chat_id)
        except Exception as e:
            logger.error("Telegram command %s error: %s", cmd, e)
            await send_message(
                f"명령어 실행 실패: {e}",
                chat_id,
                category="error",
                caller="bot._process_update",
            )
    elif text.startswith("/"):
        await send_message(
            "사용 가능한 명령어:\n"
            "/status — 워크플로우 상태\n"
            "/stop — 긴급 정지\n"
            "/resume — 정지 해제\n"
            "/skip — 오늘 스킵\n"
            "/report — 일일 리포트\n"
            "/factors — 최고 팩터",
            chat_id,
            category="command_response",
            caller="bot._process_update",
        )


async def _poll_loop() -> None:
    """Long polling으로 Telegram 업데이트 수신."""
    global _last_update_id

    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        logger.info("TELEGRAM_BOT_TOKEN 미설정 — 텔레그램 봇 비활성")
        return

    logger.info("Telegram bot polling started")
    url = f"{_BASE_URL.format(token=token)}/getUpdates"

    while True:
        try:
            async with httpx.AsyncClient(timeout=35) as client:
                params = {"offset": _last_update_id + 1, "timeout": 30}
                resp = await client.get(url, params=params)
                data = resp.json()

                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        _last_update_id = max(_last_update_id, update["update_id"])
                        try:
                            await _process_update(update)
                        except Exception as e:
                            logger.error("Telegram update error: %s", e)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Telegram poll error: %s", e)
            await asyncio.sleep(5)


async def start_bot() -> None:
    """텔레그램 봇 시작 (lifespan에서 호출).

    OpenClaw가 동일 봇 토큰으로 getUpdates 폴링을 하므로,
    백엔드에서는 수신 폴링을 하지 않는다 (409 Conflict 방지).
    send_message()를 통한 전송만 사용.
    """
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        logger.info("텔레그램 봇 설정 없음 — 스킵")
        return
    logger.info("텔레그램 봇 초기화 (전송 전용, 폴링 비활성 — OpenClaw이 수신 담당)")


async def stop_bot() -> None:
    """텔레그램 봇 중지."""
    global _poll_task
    if _poll_task and not _poll_task.done():
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass
    _poll_task = None
    logger.info("Telegram bot stopped")
