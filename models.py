"""Data models — the shared vocabulary between all layers."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Action types the planner can emit ──────────────────────────────────────────

class ActionType(str, Enum):
    LAUNCH_APP = "launch_app"
    CLICK_ELEMENT = "click_element"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    WAIT = "wait"
    FOCUS_WINDOW = "focus_window"


# ── Planner output ─────────────────────────────────────────────────────────────

class Step(BaseModel):
    """One atomic instruction in the plan."""

    id: int
    action: ActionType
    target: str = Field(
        default="",
        description="What to interact with (app name, element description, key name).",
    )
    text: str = Field(
        default="",
        description="Text to type (only for type_text action).",
    )
    key: str = Field(
        default="",
        description="Key to press (only for press_key action).",
    )
    success_condition: str = Field(
        default="",
        description="Natural-language description of what the screen should look like when done.",
    )


class Plan(BaseModel):
    """The full plan Gemini generates."""

    task: str
    steps: list[Step]


# ── Perceiver output ──────────────────────────────────────────────────────────

class UIElement(BaseModel):
    """One interactive element on screen."""

    role: str  # e.g. "push_button", "text_entry", "menu_item"
    name: str  # e.g. "Search or enter address"
    x: int
    y: int
    w: int
    h: int
    path: str = ""
    relative_x: Optional[int] = None
    relative_y: Optional[int] = None
    children: list[UIElement] = Field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class WindowInfo(BaseModel):
    """A window on screen."""

    window_class: str = Field(alias="class", default="")
    title: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    workspace: str = ""
    focused: bool = False

    model_config = {"populate_by_name": True}


class ScreenState(BaseModel):
    """Complete structured snapshot of the screen."""

    cursor_x: int = 0
    cursor_y: int = 0
    active_window: Optional[WindowInfo] = None
    windows: list[WindowInfo] = Field(default_factory=list)
    ui_tree: list[UIElement] = Field(default_factory=list)


# ── Session Logging Models ─────────────────────────────────────────────────────

class ActionRecord(BaseModel):
    """Record of a low-level desktop action."""

    action: str
    params: dict = Field(default_factory=dict)
    timestamp: str = ""
    duration_ms: float = 0.0
    status: str = "success"  # "success", "failed", "skipped"
    details: Optional[str] = None


class JevRecord(BaseModel):
    """Record of a Jev interaction (state sent, questions asked, answers received)."""

    timestamp: str = ""
    model: str = ""
    state_payload: dict = Field(default_factory=dict)
    questions: dict = Field(default_factory=dict)
    response: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class AttemptRecord(BaseModel):
    """Record of an attempt within a step."""

    attempt_number: int
    timestamp: str = ""
    screen_state: Optional[dict] = None
    jev: Optional[JevRecord] = None
    action: Optional[ActionRecord] = None
    step_complete: bool = False
    notes: str = ""


class StepExecutionRecord(BaseModel):
    """Record of a step's full execution lifecycle."""

    step_id: int
    action: str
    target: str = ""
    text: str = ""
    key: str = ""
    success_condition: str = ""
    status: str = "pending"  # "pending", "completed", "failed", "skipped"
    attempts: list[AttemptRecord] = Field(default_factory=list)
    error: Optional[str] = None


class LLMInteractionRecord(BaseModel):
    """Record of planner LLM prompt and response."""

    model: str = ""
    prompt: str = ""
    system_prompt: str = ""
    raw_response: Optional[str] = None
    parsed_plan: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class SessionSummary(BaseModel):
    """Aggregated stats for the session."""

    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    total_actions: int = 0
    total_jev_calls: int = 0


class SessionLog(BaseModel):
    """Complete record of an execution session."""

    session_id: str
    task: str
    created_at: str
    ended_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "running"  # "running", "success", "failed", "error", "cancelled"
    error: Optional[str] = None
    llm: Optional[LLMInteractionRecord] = None
    steps: list[StepExecutionRecord] = Field(default_factory=list)
    summary: SessionSummary = Field(default_factory=SessionSummary)

