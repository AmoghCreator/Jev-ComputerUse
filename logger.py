"""Session-based logging mechanism for Jev Computer Use Agent.

Records:
  1. Desktop actions performed (ydotool, hyprctl, parameters, timing, status).
  2. Perceived screen state passed (cursor coords, active window, windows, UI elements).
  3. Jev payloads and responses (state sent, question definitions, probabilities, choices).
  4. LLM interactions (Gemini task prompt, system instruction, raw response, parsed plan).
"""

from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import config
from models import (
    ActionRecord,
    AttemptRecord,
    JevRecord,
    LLMInteractionRecord,
    Plan,
    ScreenState,
    SessionLog,
    SessionSummary,
    Step,
    StepExecutionRecord,
)

# Global active session instance
_ACTIVE_LOGGER: Optional[SessionLogger] = None


def get_logger() -> Optional[SessionLogger]:
    """Get the current active session logger, if any."""
    return _ACTIVE_LOGGER


class SessionLogger:
    """Manages session logs in JSON and formatted text files."""

    def __init__(
        self,
        task: str,
        session_id: Optional[str] = None,
        logs_dir: Optional[str] = None,
    ) -> None:
        self.task = task
        self.logs_base_dir = Path(logs_dir or config.LOGS_DIR)
        self.sessions_dir = self.logs_base_dir / "sessions"

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]
        self.session_id = session_id or f"session_{timestamp_str}_{unique_suffix}"

        self.session_dir = self.sessions_dir / self.session_id
        self.json_file = self.session_dir / "session.json"
        self.log_file = self.session_dir / "session.log"

        now_iso = datetime.datetime.now().isoformat()
        self.session_log = SessionLog(
            session_id=self.session_id,
            task=task,
            created_at=now_iso,
            status="running",
        )
        self._start_time = time.time()

        if config.ENABLE_SESSION_LOGGING:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._write_text_log(f"=== Session Started: {self.session_id} ===")
            self._write_text_log(f"Task: {task}")
            self._write_text_log(f"Started at: {now_iso}\n")
            self._update_latest_symlink()
            self._save_json()

    # ── Symlink and File Helpers ───────────────────────────────────────────────

    def _update_latest_symlink(self) -> None:
        """Create or update the 'logs/latest' symlink."""
        try:
            latest_link = self.logs_base_dir / "latest"
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            # Relative target for portability
            target = Path("sessions") / self.session_id
            latest_link.symlink_to(target, target_is_directory=True)
        except Exception:
            pass

    def _write_text_log(self, message: str) -> None:
        """Append a message to session.log."""
        if not config.ENABLE_SESSION_LOGGING:
            return
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass

    def _save_json(self) -> None:
        """Atomically persist session.json."""
        if not config.ENABLE_SESSION_LOGGING:
            return
        try:
            # Update summary metrics
            total_steps = len(self.session_log.steps)
            completed_steps = sum(
                1
                for s in self.session_log.steps
                if s.status in ("completed", "success")
            )
            failed_steps = sum(
                1
                for s in self.session_log.steps
                if s.status in ("failed", "error")
            )
            total_actions = 0
            total_jev_calls = 0

            for s in self.session_log.steps:
                for att in s.attempts:
                    if att.action:
                        total_actions += 1
                    if att.jev:
                        total_jev_calls += 1

            self.session_log.summary = SessionSummary(
                total_steps=total_steps,
                completed_steps=completed_steps,
                failed_steps=failed_steps,
                total_actions=total_actions,
                total_jev_calls=total_jev_calls,
            )

            # Write formatted JSON
            temp_path = self.json_file.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(self.session_log.model_dump_json(indent=2))
            temp_path.replace(self.json_file)
        except Exception:
            pass

    # ── LLM Logging ────────────────────────────────────────────────────────────

    def log_llm(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        raw_response: Optional[str] = None,
        parsed_plan: Optional[Plan | dict] = None,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Log the Gemini LLM prompt and response."""
        plan_dict = None
        if parsed_plan:
            plan_dict = (
                parsed_plan.model_dump()
                if hasattr(parsed_plan, "model_dump")
                else parsed_plan
            )

        self.session_log.llm = LLMInteractionRecord(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            raw_response=raw_response,
            parsed_plan=plan_dict,
            duration_ms=round(duration_ms, 2),
            error=error,
        )

        self._write_text_log(f"[LLM] Model: {model} (took {duration_ms:.1f}ms)")
        if error:
            self._write_text_log(f"[LLM ERROR] {error}")
        else:
            step_count = len(plan_dict.get("steps", [])) if plan_dict else 0
            self._write_text_log(f"[LLM] Generated plan with {step_count} steps")

        self._save_json()

    # ── Plan & Step Lifecycle ──────────────────────────────────────────────────

    def init_plan(self, plan: Plan) -> None:
        """Initialize the plan steps in the session log."""
        self.session_log.steps = []
        for step in plan.steps:
            self.session_log.steps.append(
                StepExecutionRecord(
                    step_id=step.id,
                    action=step.action.value,
                    target=step.target,
                    text=step.text,
                    key=step.key,
                    success_condition=step.success_condition,
                    status="pending",
                )
            )
        self._write_text_log(f"[PLAN] Initialized {len(plan.steps)} steps")
        self._save_json()

    def start_step(self, step: Step) -> None:
        """Mark a step as started."""
        record = self._get_step_record(step.id)
        if not record:
            record = StepExecutionRecord(
                step_id=step.id,
                action=step.action.value,
                target=step.target,
                text=step.text,
                key=step.key,
                success_condition=step.success_condition,
                status="pending",
            )
            self.session_log.steps.append(record)

        self._write_text_log(
            f"--- Step {step.id}: {step.action.value} --- "
            f"Target: '{step.target or step.text or step.key}' "
            f"| Condition: '{step.success_condition}'"
        )
        self._save_json()

    def _get_step_record(self, step_id: int) -> Optional[StepExecutionRecord]:
        for s in self.session_log.steps:
            if s.step_id == step_id:
                return s
        return None

    # ── Jev & State & Action Logging ───────────────────────────────────────────

    def record_attempt(
        self,
        step_id: int,
        attempt_number: int,
        screen_state: Optional[ScreenState | dict] = None,
        jev_payload: Optional[dict] = None,
        jev_questions: Optional[dict] = None,
        jev_response: Optional[Any] = None,
        jev_duration_ms: float = 0.0,
        action_record: Optional[ActionRecord] = None,
        step_complete: bool = False,
        notes: str = "",
        error: Optional[str] = None,
    ) -> None:
        """Record an evaluation cycle / retry attempt within a step."""
        step_rec = self._get_step_record(step_id)
        if not step_rec:
            return

        now_iso = datetime.datetime.now().isoformat()

        # Serialize screen state
        screen_dict = None
        if screen_state:
            screen_dict = (
                screen_state.model_dump()
                if hasattr(screen_state, "model_dump")
                else screen_state
            )

        # Serialize questions
        questions_dict = {}
        if jev_questions:
            for k, q in jev_questions.items():
                if hasattr(q, "model_dump"):
                    questions_dict[k] = q.model_dump()
                elif isinstance(q, dict):
                    questions_dict[k] = q
                else:
                    questions_dict[k] = str(q)

        # Serialize response
        response_dict = None
        if jev_response:
            if hasattr(jev_response, "model_dump"):
                response_dict = jev_response.model_dump()
            elif isinstance(jev_response, dict):
                response_dict = jev_response
            else:
                response_dict = {"raw": str(jev_response)}

        jev_rec = None
        if jev_payload or jev_questions or jev_response or error:
            jev_rec = JevRecord(
                timestamp=now_iso,
                model=config.JEV_MODEL,
                state_payload=jev_payload or {},
                questions=questions_dict,
                response=response_dict,
                error=error,
                duration_ms=round(jev_duration_ms, 2),
            )

        att_rec = AttemptRecord(
            attempt_number=attempt_number,
            timestamp=now_iso,
            screen_state=screen_dict,
            jev=jev_rec,
            action=action_record,
            step_complete=step_complete,
            notes=notes,
        )

        step_rec.attempts.append(att_rec)

        # Format text log entries
        self._write_text_log(f"  [Attempt {attempt_number}]")
        if screen_dict:
            active = screen_dict.get("active_window") or {}
            active_str = f"'{active.get('class','')}' - '{active.get('title','')}'"
            self._write_text_log(
                f"    Perceiver State: Active={active_str}, "
                f"Cursor=({screen_dict.get('cursor_x', 0)}, {screen_dict.get('cursor_y', 0)}), "
                f"UI Elements={len(screen_dict.get('ui_elements', []))}"
            )

        if jev_rec:
            ui_count = 0
            if "raw_state" in jev_payload:
                ui_count = jev_payload["raw_state"].count("role:")
            else:
                ui_count = len(jev_payload.get('ui_elements', []))
            self._write_text_log(
                f"    Jev Payload: sent {ui_count} elements, "
                f"asked {list(questions_dict.keys())} (took {jev_duration_ms:.1f}ms)"
            )
            if response_dict and "answers" in response_dict:
                ans_str = ", ".join(
                    f"{k}={v.get('choice') if 'choice' in v else v.get('noul')}"
                    for k, v in response_dict["answers"].items()
                )
                self._write_text_log(f"    Jev Response: {ans_str}")
            elif error:
                self._write_text_log(f"    Jev Error: {error}")

        if action_record:
            self._write_text_log(
                f"    Action Executed: {action_record.action} "
                f"params={action_record.params} (status={action_record.status}, "
                f"took {action_record.duration_ms:.1f}ms)"
            )

        self._save_json()

    def record_direct_action(self, step_id: int, action_record: ActionRecord) -> None:
        """Record a direct deterministic action (e.g. launch, key press, type)."""
        step_rec = self._get_step_record(step_id)
        if not step_rec:
            return

        now_iso = datetime.datetime.now().isoformat()
        att_rec = AttemptRecord(
            attempt_number=1,
            timestamp=now_iso,
            action=action_record,
            step_complete=True,
            notes="Direct deterministic execution",
        )
        step_rec.attempts.append(att_rec)

        self._write_text_log(
            f"  [Direct Action] {action_record.action}: {action_record.params} "
            f"({action_record.status}, {action_record.duration_ms:.1f}ms)"
        )
        self._save_json()

    def complete_step(
        self, step_id: int, status: str = "completed", error: Optional[str] = None
    ) -> None:
        """Mark a step as finished."""
        if status in ("success", "completed"):
            norm_status = "completed"
        elif status in ("failed", "error"):
            norm_status = "failed"
        elif status in ("skipped",):
            norm_status = "skipped"
        else:
            norm_status = status

        step_rec = self._get_step_record(step_id)
        if step_rec:
            step_rec.status = norm_status
            step_rec.error = error

        symbol = "✓" if norm_status == "completed" else "✗"
        self._write_text_log(
            f"  {symbol} Step {step_id} {norm_status.upper()}"
            + (f": {error}" if error else "")
        )
        self._save_json()


    def end_session(
        self, status: str = "success", error: Optional[str] = None
    ) -> None:
        """Finalize the session, calculate durations and write summary."""
        ended_at = datetime.datetime.now().isoformat()
        duration = round(time.time() - self._start_time, 2)

        self.session_log.ended_at = ended_at
        self.session_log.duration_seconds = duration
        self.session_log.status = status
        self.session_log.error = error

        self._write_text_log(
            f"\n=== Session Finished: {status.upper()} in {duration}s ==="
        )
        if error:
            self._write_text_log(f"Error: {error}")

        self._save_json()


# ── Global Lifecycle Functions ─────────────────────────────────────────────────

def start_session(
    task: str,
    session_id: Optional[str] = None,
    logs_dir: Optional[str] = None,
) -> SessionLogger:
    """Initialize and set the global session logger."""
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = SessionLogger(
        task=task, session_id=session_id, logs_dir=logs_dir
    )
    return _ACTIVE_LOGGER


def end_session(status: str = "success", error: Optional[str] = None) -> None:
    """End the global session logger if active."""
    global _ACTIVE_LOGGER
    if _ACTIVE_LOGGER:
        _ACTIVE_LOGGER.end_session(status=status, error=error)
        _ACTIVE_LOGGER = None


# ── Inspection & Reporting Utilities ───────────────────────────────────────────

def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """List recent sessions found in the logs directory."""
    sessions_dir = Path(config.LOGS_DIR) / "sessions"
    if not sessions_dir.exists():
        return []

    results = []
    # Sorted by folder modification time, descending
    dirs = sorted(
        [d for d in sessions_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for d in dirs[:limit]:
        json_file = d / "session.json"
        if json_file.exists():
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(
                        {
                            "session_id": data.get("session_id", d.name),
                            "task": data.get("task", ""),
                            "status": data.get("status", "unknown"),
                            "created_at": data.get("created_at", ""),
                            "duration_seconds": data.get("duration_seconds", 0),
                            "steps_total": data.get("summary", {}).get(
                                "total_steps", 0
                            ),
                            "steps_completed": data.get("summary", {}).get(
                                "completed_steps", 0
                            ),
                            "actions_total": data.get("summary", {}).get(
                                "total_actions", 0
                            ),
                            "jev_calls": data.get("summary", {}).get(
                                "total_jev_calls", 0
                            ),
                        }
                    )
            except Exception:
                results.append({"session_id": d.name, "status": "corrupt"})
    return results


def load_session(session_id: str = "latest") -> Optional[SessionLog]:
    """Load a session log by ID or 'latest'."""
    logs_base = Path(config.LOGS_DIR)
    if session_id == "latest":
        target = logs_base / "latest" / "session.json"
        if not target.exists():
            sessions = list_sessions(limit=1)
            if sessions:
                target = (
                    logs_base / "sessions" / sessions[0]["session_id"] / "session.json"
                )
            else:
                return None
    else:
        target = logs_base / "sessions" / session_id / "session.json"

    if not target.exists():
        return None

    try:
        with open(target, encoding="utf-8") as f:
            return SessionLog.model_validate_json(f.read())
    except Exception as e:
        print(f"Error reading session file {target}: {e}")
        return None


def format_session_summary(session: SessionLog) -> str:
    """Format a SessionLog into a clear, detailed terminal summary."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  SESSION REPORT: {session.session_id}")
    lines.append(f"  Status: {session.status.upper()} | Duration: {session.duration_seconds or 0}s")
    lines.append(f"  Task: {session.task}")
    lines.append(f"  Created: {session.created_at}")
    lines.append("=" * 70)

    # LLM Interaction
    if session.llm:
        lines.append("\n[1] LLM Interaction (Gemini Planner)")
        lines.append(f"    Model: {session.llm.model} ({session.llm.duration_ms}ms)")
        if session.llm.error:
            lines.append(f"    Error: {session.llm.error}")
        elif session.llm.parsed_plan:
            steps = session.llm.parsed_plan.get("steps", [])
            lines.append(f"    Plan generated ({len(steps)} steps):")
            for st in steps:
                lines.append(
                    f"      - Step {st.get('id')}: [{st.get('action')}] "
                    f"target='{st.get('target') or st.get('text') or st.get('key')}'"
                )
                if st.get("success_condition"):
                    lines.append(f"        Success: {st.get('success_condition')}")

    # Steps Execution
    lines.append(f"\n[2] Execution Trace ({len(session.steps)} steps)")
    for step in session.steps:
        status_icon = "✓" if step.status == "completed" else "✗"
        lines.append(
            f"\n  {status_icon} Step {step.step_id} [{step.action}] "
            f"target='{step.target or step.text or step.key}' "
            f"Status: {step.status.upper()}"
        )
        if step.success_condition:
            lines.append(f"    Goal: {step.success_condition}")

        for att in step.attempts:
            lines.append(f"    Attempt #{att.attempt_number} ({att.timestamp}):")
            if att.screen_state:
                act = att.screen_state.get("active_window") or {}
                lines.append(
                    f"      Screen State: Active='{act.get('title','')}' "
                    f"({act.get('class','')}), "
                    f"Cursor=({att.screen_state.get('cursor_x')}, {att.screen_state.get('cursor_y')}), "
                    f"UI Elements={len(att.screen_state.get('ui_elements', []))}"
                )

            if att.jev:
                ui_count = 0
                if "raw_state" in att.jev.state_payload:
                    ui_count = att.jev.state_payload["raw_state"].count("role:")
                else:
                    ui_count = len(att.jev.state_payload.get('ui_elements', []))
                lines.append(
                    f"      Jev Payload: {ui_count} elements passed, "
                    f"Questions={list(att.jev.questions.keys())} ({att.jev.duration_ms}ms)"
                )
                if att.jev.response and "answers" in att.jev.response:
                    ans = att.jev.response["answers"]
                    parts = []
                    if "step_complete" in ans:
                        prob = ans["step_complete"].get("noul", 0.0)
                        parts.append(f"complete_prob={prob:.3f}")
                    if "target_element" in ans:
                        choice = ans["target_element"].get("choice", "")
                        parts.append(f"target_element='{choice}'")
                    lines.append(f"      Jev Decision: {', '.join(parts)}")
                elif att.jev.error:
                    lines.append(f"      Jev Error: {att.jev.error}")

            if att.action:
                lines.append(
                    f"      Action Executed: {att.action.action} "
                    f"params={att.action.params} (took {att.action.duration_ms}ms, {att.action.status})"
                )

    # Summary
    lines.append("\n" + "-" * 70)
    lines.append(
        f"  Summary: {session.summary.completed_steps}/{session.summary.total_steps} steps completed | "
        f"{session.summary.total_actions} actions | {session.summary.total_jev_calls} Jev calls"
    )
    lines.append("-" * 70)

    return "\n".join(lines)
