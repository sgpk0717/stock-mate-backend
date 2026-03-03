"""에이전트 대화 REST API 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.manager import process_message
from app.agents.session import session_store
from app.backtest.schemas import StrategySchema

router = APIRouter(prefix="/agents", tags=["agents"])


# ── 요청/응답 스키마 ──


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    created_at: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    role: str
    content: str
    timestamp: str
    strategy_draft: dict | None = None


class SessionResponse(BaseModel):
    session_id: str
    messages: list[dict]
    strategy_draft: dict | None = None
    status: str
    created_at: str
    updated_at: str


class FinalizeResponse(BaseModel):
    strategy: StrategySchema
    message: str


# ── 엔드포인트 ──


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session():
    """새 대화 세션을 생성한다."""
    session = session_store.create()
    return CreateSessionResponse(
        session_id=session.id,
        status=session.status,
        created_at=session.created_at,
    )


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, req: ChatRequest):
    """메시지를 전송하고 에이전트 응답을 수신한다."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없거나 만료되었습니다.")
    if session.status == "finalized":
        raise HTTPException(400, "이미 확정된 세션입니다.")

    if not req.message.strip():
        raise HTTPException(400, "메시지가 비어있습니다.")

    try:
        assistant_msg = await process_message(session, req.message)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"에이전트 처리 실패: {e}")

    return ChatResponse(
        role=assistant_msg.role,
        content=assistant_msg.content,
        timestamp=assistant_msg.timestamp,
        strategy_draft=assistant_msg.strategy_draft,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """세션 상태를 조회한다."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없거나 만료되었습니다.")
    data = session.to_dict()
    return SessionResponse(**data)


@router.post("/sessions/{session_id}/finalize", response_model=FinalizeResponse)
async def finalize_strategy(session_id: str):
    """전략을 확정한다. strategy_draft가 있어야 한다."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없거나 만료되었습니다.")
    if session.status == "finalized":
        raise HTTPException(400, "이미 확정된 세션입니다.")
    if session.strategy_draft is None:
        raise HTTPException(
            400,
            "전략 초안이 없습니다. 대화를 통해 먼저 전략을 생성해주세요.",
        )

    try:
        strategy = StrategySchema(**session.strategy_draft)
    except Exception as e:
        raise HTTPException(400, f"전략 스키마 변환 실패: {e}")

    session.status = "finalized"
    session.touch()

    return FinalizeResponse(
        strategy=strategy,
        message="전략이 확정되었습니다. 이제 백테스트를 실행할 수 있습니다.",
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """세션을 삭제한다."""
    if not session_store.delete(session_id):
        raise HTTPException(404, "세션을 찾을 수 없습니다.")
