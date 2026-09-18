"""Unit test for session-based logging mechanism."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from models import (
    ActionRecord,
    ActionType,
    Plan,
    ScreenState,
    Step,
    UIElement,
    WindowInfo,
)
import logger
from logger import SessionLogger, list_sessions, load_session, format_session_summary


def test_session_logger():
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Start session
        sl = SessionLogger(
            task="Open calculator and compute 42 * 2",
            session_id="test_session_001",
            logs_dir=temp_dir,
        )
        assert sl.session_id == "test_session_001"
        assert sl.json_file.exists()
        assert sl.log_file.exists()

        # 2. Log LLM Interaction
        plan = Plan(
            task="Open calculator and compute 42 * 2",
            steps=[
                Step(
                    id=1,
                    action=ActionType.LAUNCH_APP,
                    target="gnome-calculator",
                    success_condition="Calculator window appears",
                ),
                Step(
                    id=2,
                    action=ActionType.CLICK_ELEMENT,
                    target="Button 4",
                    success_condition="Number 4 entered",
                ),
            ],
        )
        sl.log_llm(
            model="gemini-2.5-flash",
            prompt="Task: Open calculator and compute 42 * 2",
            system_prompt="You are an automation planner...",
            raw_response='{"task": "...", "steps": [...]}',
            parsed_plan=plan,
            duration_ms=450.5,
        )

        # 3. Init plan & start step 1
        sl.init_plan(plan)
        sl.start_step(plan.steps[0])

        # Step 1: Direct action
        act1 = ActionRecord(
            action="launch_app",
            params={"app_name": "gnome-calculator"},
            timestamp="2026-09-18T23:55:00",
            duration_ms=2100.0,
            status="success",
        )
        sl.record_direct_action(step_id=1, action_record=act1)
        sl.complete_step(step_id=1, status="completed")

        # Step 2: Jev evaluation loop
        sl.start_step(plan.steps[1])
        screen = ScreenState(
            cursor_x=350,
            cursor_y=420,
            active_window=WindowInfo(
                **{"class": "org.gnome.Calculator", "title": "Calculator", "focused": True}
            ),
            windows=[
                WindowInfo(**{"class": "org.gnome.Calculator", "title": "Calculator", "focused": True})
            ],
            ui_elements=[
                UIElement(role="push_button", name="4", x=300, y=400, w=50, h=40),
                UIElement(role="push_button", name="2", x=360, y=400, w=50, h=40),
            ],
        )
        jev_payload = {
            "current_step": {"id": 2, "action": "click_element", "target": "Button 4"},
            "cursor": {"x": 350, "y": 420},
            "ui_elements": [{"index": 0, "role": "push_button", "name": "4"}],
        }
        jev_questions = {
            "step_complete": {"type": "noul", "instructions": "Is step complete?"},
            "target_element": {
                "type": "choice",
                "instructions": "Which button?",
                "criteria": {"element_0": "4", "none": "none"},
            },
        }
        jev_response = {
            "model": "jev-latest",
            "answers": {
                "step_complete": {"type": "noul", "noul": 0.15},
                "target_element": {"type": "choice", "choice": "element_0"},
            },
        }
        act2 = ActionRecord(
            action="mouse_click",
            params={"x": 325, "y": 420, "button": "left"},
            timestamp="2026-09-18T23:55:02",
            duration_ms=45.0,
            status="success",
        )
        sl.record_attempt(
            step_id=2,
            attempt_number=1,
            screen_state=screen,
            jev_payload=jev_payload,
            jev_questions=jev_questions,
            jev_response=jev_response,
            jev_duration_ms=62.3,
            action_record=act2,
            step_complete=False,
            notes="Clicked button 4",
        )
        sl.complete_step(step_id=2, status="completed")

        # End session
        sl.end_session(status="success")

        # 4. Verify session.json contents
        with open(sl.json_file) as f:
            data = json.load(f)

        assert data["session_id"] == "test_session_001"
        assert data["status"] == "success"
        assert data["llm"]["model"] == "gemini-2.5-flash"
        assert data["llm"]["parsed_plan"]["task"] == "Open calculator and compute 42 * 2"
        assert len(data["steps"]) == 2
        assert data["summary"]["completed_steps"] == 2
        assert data["summary"]["total_actions"] == 2
        assert data["summary"]["total_jev_calls"] == 1

        # Check Jev record details in attempt
        step2_attempts = data["steps"][1]["attempts"]
        assert len(step2_attempts) == 1
        assert step2_attempts[0]["jev"]["response"]["answers"]["target_element"]["choice"] == "element_0"
        assert step2_attempts[0]["screen_state"]["cursor_x"] == 350
        assert step2_attempts[0]["action"]["action"] == "mouse_click"

        # 5. Verify format_session_summary
        loaded_log = SessionLogger.load_session if hasattr(SessionLogger, "load_session") else None
        text_summary = format_session_summary(sl.session_log)
        assert "SESSION REPORT: test_session_001" in text_summary
        assert "LLM Interaction" in text_summary
        assert "Execution Trace" in text_summary
        assert "Jev Payload" in text_summary
        assert "complete_prob=0.150" in text_summary

        print("✓ All SessionLogger unit tests passed!")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_session_logger()
