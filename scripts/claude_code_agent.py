"""Claude Code Agent SDK 래퍼 — OpenClaw이 호출하는 코드 수정 도구.

사용법:
  python scripts/claude_code_agent.py --task "버그 수정 설명" --files "file1.py,file2.tsx"

설계서 §10: Claude Agent SDK로 멀티턴 코드 수정 세션 관리.
SDK 미설치 시 CLI `-p --resume` 폴백.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

# 작업 범위 제한 (설계서 §10.2)
ALLOWED_PATHS = [
    "stock-mate-backend/",
    "stock-mate-frontend/",
    "stock-mate-data-pump/",
]
MAX_AUTO_LINES = 50


async def fix_code(task_description: str, related_files: list[str]) -> dict:
    """Claude Agent SDK로 코드 수정 실행."""
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions
    except ImportError:
        return await _cli_fallback(task_description)

    prompt = f"""
[컨텍스트]
Stock Mate 자동매매 시스템에서 발견된 문제입니다.

[문제]
{task_description}

[관련 파일]
{', '.join(related_files) if related_files else '자율 탐색'}

[제약사항]
- CLAUDE.md의 코딩 컨벤션을 따를 것
- 기존 코드를 최소한으로 변경할 것
- 수정 후 git commit 할 것 (커밋 메시지에 문제 설명 포함)
- git push는 하지 말 것
- 환경변수/리스크 파라미터를 수정하지 말 것
"""

    result = await query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
            cwd=PROJECT_ROOT,
            max_turns=20,
        ),
    )

    return {
        "success": True,
        "session_id": result.session_id,
        "output": result.text[:500],
    }


async def _cli_fallback(task_description: str) -> dict:
    """Agent SDK 미설치 시 CLI -p 폴백."""
    import subprocess

    cmd = [
        "claude",
        "-p",
        task_description,
        "--output-format",
        "json",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=PROJECT_ROOT
        )
    except FileNotFoundError:
        return {"success": False, "error": "claude CLI not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "CLI timeout (300s)"}

    if proc.returncode != 0:
        return {"success": False, "error": proc.stderr[:500]}

    try:
        data = json.loads(proc.stdout)
        return {
            "success": True,
            "session_id": data.get("session_id"),
            "output": data.get("result", "")[:500],
        }
    except json.JSONDecodeError:
        return {"success": True, "output": proc.stdout[:500]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code Agent SDK Wrapper")
    parser.add_argument("--task", required=True, help="수정할 작업 설명")
    parser.add_argument("--files", default="", help="관련 파일 (쉼표 구분)")
    args = parser.parse_args()

    files = [f.strip() for f in args.files.split(",") if f.strip()]
    result = asyncio.run(fix_code(args.task, files))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
