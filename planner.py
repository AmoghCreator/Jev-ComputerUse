"""Planner — uses Gemini to generate a structured step-by-step plan.

Called ONCE at the start. Takes the user's natural language task and
produces a Plan (list of Steps) that Jev can execute.
"""

from __future__ import annotations

import time

from google import genai

import config
from logger import get_logger
from models import Plan


_SYSTEM_PROMPT = """\
You are a computer automation planner. The user gives you a task they want \
performed on a Linux desktop (Hyprland/Wayland). You must break it down into \
a precise sequence of atomic steps.

Each step is one of these action types:
- launch_app: Open an application. Set "target" to the executable name.
- click_element: Click a UI element. Set "target" to a description of what to click.
- type_text: Type text. Set "text" to the string to type.
- press_key: Press a key combo. Set "key" to the combo (e.g. "Return", "ctrl+a", "super").
- scroll: Scroll. Set "target" to "up" or "down".
- wait: Wait for the screen to change. Set "target" to what to wait for.
- focus_window: Focus a specific window. Set "target" to the window class or name.

Rules:
- Be precise and atomic. One action per step.
- Always include a success_condition describing what the screen should look like after.
- Prefer keyboard shortcuts over clicking when possible (faster, more reliable).
- In web browsers, ALWAYS press 'ctrl+l' to focus the address bar before typing a URL, followed by pressing 'Return'.
- To read or view a specific link or topic on a page, use click_element targeting that topic or link title.
- For launching apps, use the actual executable name (e.g. "firefox", "nautilus", "alacritty").
- For key combos, use format like "ctrl+l", "alt+f4", "super", "Return".
- Number steps starting from 1.
"""



def generate_plan(task: str) -> Plan:
    """Call Gemini to generate an execution plan for the given task."""
    logger = get_logger()
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    t0 = time.time()
    prompt = f"Task: {task}"

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": _SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": Plan,
            },
        )

        duration_ms = (time.time() - t0) * 1000
        plan = response.parsed
        raw_text = getattr(response, "text", "")

        if plan is None:
            err_msg = f"Gemini returned no parsed plan. Raw: {raw_text}"
            if logger:
                logger.log_llm(
                    model=config.GEMINI_MODEL,
                    prompt=prompt,
                    system_prompt=_SYSTEM_PROMPT,
                    raw_response=raw_text,
                    parsed_plan=None,
                    duration_ms=duration_ms,
                    error=err_msg,
                )
            raise RuntimeError(err_msg)

        if logger:
            logger.log_llm(
                model=config.GEMINI_MODEL,
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                raw_response=raw_text,
                parsed_plan=plan,
                duration_ms=duration_ms,
            )

        return plan

    except Exception as e:
        duration_ms = (time.time() - t0) * 1000
        if logger and not (logger.session_log.llm and logger.session_log.llm.error):
            logger.log_llm(
                model=config.GEMINI_MODEL,
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                raw_response=None,
                parsed_plan=None,
                duration_ms=duration_ms,
                error=str(e),
            )
        raise

