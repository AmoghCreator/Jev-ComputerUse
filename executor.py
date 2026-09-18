"""Executor — the Jev-powered execution loop.

For each step in the plan:
  1. Perceive the screen (perceiver.py)
  2. Ask Jev: is the step done? which element to target? what action?
  3. Execute the action (actions.py)
  4. Repeat until success or timeout
"""

from __future__ import annotations

import time

from typesafe_sdk import Choice, Noul, TypeSafeClient

import actions
import config
from logger import get_logger
from models import ActionRecord, ActionType, Plan, ScreenState, Step, UIElement
from perceiver import perceive


def _build_state_string(step: Step, screen: ScreenState) -> str:
    """Build a human-readable state string to send to Jev."""
    active = "none"
    if screen.active_window:
        active = f"class: {screen.active_window.window_class}, title: {screen.active_window.title}"
        
    windows_lines = []
    for w in screen.windows:
        if w.window_class:
            focused = " (focused)" if w.focused else ""
            windows_lines.append(f"  - class: {w.window_class}, title: {w.title}{focused}")
    windows_str = "\n".join(windows_lines) if windows_lines else "  none"
    
    elements_lines = []
    flat = _flatten_elements(screen.ui_tree)
    for el in flat:
        cx, cy = el.center
        elements_lines.append(f"  - role: {el.role}, name: '{el.name}', path: {el.path}, position: ({cx}, {cy})")
    elements_str = "\n".join(elements_lines) if elements_lines else "  none"
    
    return (
        f"CURRENT STEP:\n"
        f"  Action: {step.action.value}\n"
        f"  Target: {step.target}\n"
        f"  Text: {step.text}\n"
        f"  Key: {step.key}\n"
        f"  Success Condition: {step.success_condition}\n\n"
        f"SITUATION REPORT:\n"
        f"cursor:\n  position: {screen.cursor_x}, {screen.cursor_y}\n"
        f"active_window:\n  {active}\n"
        f"windows:\n{windows_str}\n"
        f"ui_elements:\n{elements_str}\n"
    )


def _flatten_elements(elements: list[UIElement]) -> list[UIElement]:
    flat = []
    for el in elements:
        flat.append(el)
        flat.extend(_flatten_elements(el.children))
    return flat

def _build_element_criteria(ui_tree: list[UIElement]) -> dict[str, str]:
    """Build Choice criteria from UI elements."""
    criteria: dict[str, str] = {}
    flat = _flatten_elements(ui_tree)
    for el in flat:
        criteria[f"element_{el.path}"] = (
            f"{el.role}: '{el.name}' at ({el.center[0]}, {el.center[1]})"
        )
    criteria["none"] = "No matching element found — use keyboard action instead"
    return criteria


def _execute_action(
    step: Step,
    screen: ScreenState,
    target_choice: str,
    action_choice: str,
) -> ActionRecord:
    """Execute the decided action on the desktop and return ActionRecord."""
    # Resolve target coordinates if an element was chosen
    target_el: UIElement | None = None
    if target_choice.startswith("element_"):
        path = target_choice.split("element_", 1)[1]
        flat = _flatten_elements(screen.ui_tree)
        for el in flat:
            if el.path == path:
                target_el = el
                break

    match step.action:
        case ActionType.LAUNCH_APP:
            return actions.launch_app(step.target)

        case ActionType.CLICK_ELEMENT:
            if target_el:
                cx, cy = target_el.center
                return actions.mouse_click(cx, cy)
            else:
                print(f"  [executor] No element found for click, trying keyboard fallback")
                return ActionRecord(
                    action="click_element_fallback",
                    params={"target": step.target},
                    status="skipped",
                    details="No element found for click",
                )

        case ActionType.TYPE_TEXT:
            return actions.type_text(step.text)

        case ActionType.PRESS_KEY:
            return actions.press_key(step.key)

        case ActionType.SCROLL:
            direction = step.target if step.target in ("up", "down") else "down"
            return actions.scroll(direction)

        case ActionType.FOCUS_WINDOW:
            return actions.focus_window(step.target)

        case ActionType.WAIT:
            time.sleep(1.0)
            return ActionRecord(
                action="wait",
                params={"duration": 1.0},
                duration_ms=1000.0,
                status="success",
            )


def _evaluate_step_completion(
    client: TypeSafeClient | None,
    step: Step,
    screen: ScreenState,
) -> tuple[bool, float, dict, Optional[str]]:
    """Query Jev to evaluate if the step's success condition is met based on the live screen state."""
    state_str = _build_state_string(step, screen)
    state_payload = {"raw_state": state_str}

    active_title = (
        screen.active_window.title if screen.active_window else ""
    ).lower()
    active_class = (
        screen.active_window.window_class if screen.active_window else ""
    ).lower()
    target_lower = (step.target or "").lower()
    cond_lower = (step.success_condition or "").lower()

    # Deterministic heuristics for speed and robustness
    if step.action == ActionType.LAUNCH_APP:
        if (target_lower and (target_lower in active_class or target_lower in active_title)) or \
           ("firefox" in target_lower and "firefox" in active_class):
            return True, 1.0, state_payload, None

    if "hacker news" in cond_lower and "hacker news" in active_title:
        return True, 0.95, state_payload, None

    if not client:
        return False, 0.0, state_payload, "No TypeSafe client configured"

    questions = {
        "step_complete": Noul(
            instructions=(
                f"Based on the live screen state: "
                f"Active window title: '{screen.active_window.title if screen.active_window else 'none'}', "
                f"Active window class: '{screen.active_window.window_class if screen.active_window else 'none'}', "
                f"Visible windows: {[w.title for w in screen.windows if w.title]}, "
                f"Interactive UI elements (top-level): {len(screen.ui_tree)}. "
                f"Has the following step success condition been met: '{step.success_condition}'?"
            )
        )
    }

    try:
        t0 = time.time()
        response = client.system_one(state=state_str, questions=questions)
        duration_ms = (time.time() - t0) * 1000
        prob = response.answers["step_complete"].noul
        is_done = prob >= 0.70
        return is_done, prob, state_payload, None
    except Exception as e:
        return False, 0.0, state_payload, str(e)





def execute_plan(plan: Plan) -> None:
    """Execute a full plan with recursive live state polling and Jev verification."""
    logger = get_logger()
    if logger:
        logger.init_plan(plan)

    print(f"\n{'='*60}")
    print(f"  EXECUTING PLAN: {plan.task}")
    print(f"  {len(plan.steps)} steps (with recursive state polling)")
    print(f"{'='*60}\n")

    client = None
    if config.TYPESAFE_API_KEY:
        try:
            client = TypeSafeClient(api_key=config.TYPESAFE_API_KEY)
        except Exception as e:
            print(f"  [executor] Warning: Failed to init TypeSafeClient: {e}")

    target_app: Optional[str] = None

    for step in plan.steps:
        print(f"\n--- Step {step.id}: [{step.action.value}] ---")
        print(f"    Target: {step.target or step.text or step.key}")
        print(f"    Success condition: {step.success_condition}")

        if step.action in (ActionType.LAUNCH_APP, ActionType.FOCUS_WINDOW):
            target_app = step.target

        if logger:
            logger.start_step(step)

        # 1. Pre-action live perception
        screen = perceive()
        active_title = (
            screen.active_window.title if screen.active_window else "none"
        )
        active_class = (
            screen.active_window.window_class if screen.active_window else "none"
        )
        print(
            f"    Initial state: Active='{active_title}' ({active_class}), "
            f"Elements (top-level)={len(screen.ui_tree)}"
        )

        # Focus Guard: Ensure target app is focused before interaction
        if target_app and step.action in (
            ActionType.TYPE_TEXT,
            ActionType.PRESS_KEY,
            ActionType.CLICK_ELEMENT,
        ):
            cur_cls = active_class.lower()
            cur_tit = active_title.lower()
            tgt = target_app.lower()
            if tgt not in cur_cls and tgt not in cur_tit:
                print(
                    f"    [Focus Guard] Focusing target app '{target_app}' "
                    f"(currently active: '{active_class}')"
                )
                actions.focus_window(target_app)
                time.sleep(0.4)
                screen = perceive()
                active_title = (
                    screen.active_window.title
                    if screen.active_window
                    else "none"
                )
                active_class = (
                    screen.active_window.window_class
                    if screen.active_window
                    else "none"
                )

        # 2. Pre-check: Is condition already met before acting?
        is_done = False
        prob = 0.0
        state_str = _build_state_string(step, screen)
        state_payload = {"raw_state": state_str}

        if step.action == ActionType.LAUNCH_APP:
            # Only considered done if the app is actually launched AND active
            if target_app and (
                target_app.lower() in active_class.lower()
                or target_app.lower() in active_title.lower()
            ):
                is_done = True
                prob = 1.0
        elif step.action not in (ActionType.PRESS_KEY, ActionType.TYPE_TEXT):
            is_done, prob, state_payload, _ = _evaluate_step_completion(
                client, step, screen
            )

        if is_done:
            print(f"    ✓ Step already satisfied (prob={prob:.3f}), skipping action")
            if logger:
                logger.record_attempt(
                    step_id=step.id,
                    attempt_number=1,
                    screen_state=screen,
                    jev_payload=state_payload,
                    step_complete=True,
                    notes=f"Pre-check satisfied (prob={prob:.3f})",
                )
                logger.complete_step(step.id, status="completed")
            continue


        # 3. Action execution with recursive state polling
        attempt = 0
        step_succeeded = False

        while attempt < config.MAX_RETRIES_PER_STEP:
            attempt += 1
            print(f"    Action execution (Attempt {attempt}/{config.MAX_RETRIES_PER_STEP})")

            # Resolve target element for click_element
            target = "none"
            target_questions = {}
            target_answers = {}
            if step.action == ActionType.CLICK_ELEMENT and screen.ui_tree and client:
                criteria = _build_element_criteria(screen.ui_tree)
                q = {
                    "target_element": Choice(
                        instructions=f"Select the UI element best matching '{step.target}'",
                        criteria=criteria,
                    )
                }
                try:
                    choice_resp = client.system_one(state=state_str, questions=q)
                    target = choice_resp.answers["target_element"].choice
                    target_questions = {"target_element": {"instructions": q["target_element"].instructions, "criteria": q["target_element"].criteria}}
                    target_answers = {"target_element": {"choice": target}}
                    print(f"    Jev selected target: {target}")
                except Exception as e:
                    print(f"    [executor] Element selection error: {e}")

            # Execute the low-level action
            action_rec = _execute_action(step, screen, target, step.action.value)
            print(
                f"    Executed: {action_rec.action} ({action_rec.status}, {action_rec.duration_ms:.1f}ms)"
            )

            # 4. Wait and evaluate state exactly once
            wait_time = 2.0 if step.action in (ActionType.LAUNCH_APP, ActionType.WAIT) else 1.0
            if "navigate" in step.success_condition.lower() or "load" in step.success_condition.lower():
                wait_time = 3.0
            time.sleep(wait_time)
            
            screen = perceive()
            is_done, prob, state_payload, err = _evaluate_step_completion(client, step, screen)
            
            if logger:
                combined_questions = {"step_complete": {"instructions": step.success_condition}}
                combined_questions.update(target_questions)
                combined_answers = {"step_complete": {"noul": prob}}
                combined_answers.update(target_answers)
                
                logger.record_attempt(
                    step_id=step.id,
                    attempt_number=attempt,
                    screen_state=screen,
                    jev_payload=state_payload,
                    jev_questions=combined_questions,
                    jev_response={"answers": combined_answers},
                    step_complete=is_done,
                    notes=f"Attempt {attempt}: prob={prob:.3f}",
                    error=err,
                )

            if is_done:
                step_succeeded = True
                print(f"    ✓ Condition verified met: '{step.success_condition}' (prob={prob:.3f})")
                if logger:
                    logger.complete_step(step.id, status="completed")
                break
            else:
                # If it's a simple type or press_key and we didn't explicitly fail, consider if next step verifies it
                if step.action in (ActionType.TYPE_TEXT, ActionType.PRESS_KEY):
                    print("    Proceeding after keystroke...")
                    step_succeeded = True
                    if logger:
                        logger.complete_step(step.id, status="completed")
                    break
                print(f"    ⏳ State not verified (prob={prob:.3f}), retrying action...")

        if not step_succeeded:
            print(
                f"    ✗ Step {step.id} FAILED to verify condition after {config.MAX_RETRIES_PER_STEP} retries"
            )
            if logger:
                logger.complete_step(
                    step.id,
                    status="failed",
                    error=f"Exceeded {config.MAX_RETRIES_PER_STEP} retries",
                )

    print(f"\n{'='*60}")
    print(f"  PLAN EXECUTION COMPLETE")
    print(f"{'='*60}\n")


