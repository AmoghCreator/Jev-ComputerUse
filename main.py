#!/usr/bin/env python3
"""Computer Use Agent — LLM (Gemini) + Jev (TypeSafe).

Usage:
    python main.py "open firefox and search for python tutorials"
    python main.py --plan-only "open nautilus and create a folder"
    python main.py --plan-file test_plan.json
    python main.py --perceive  # dump current screen state

Inspection:
    python main.py --list-sessions
    python main.py --show-session [latest | session_id]
"""

from __future__ import annotations

import argparse
import json
import sys

import config
from executor import execute_plan
from logger import (
    end_session,
    format_session_summary,
    list_sessions,
    load_session,
    start_session,
)
from models import Plan
from perceiver import perceive
from planner import generate_plan


def _handle_list_sessions() -> None:
    """Print a table of recent sessions."""
    sessions = list_sessions(limit=15)
    if not sessions:
        print("No sessions found in logs directory.")
        return

    print("\n" + "=" * 90)
    print(f"  {'SESSION ID':<34} {'STATUS':<10} {'STEPS':<8} {'ACTIONS':<8} {'TASK':<25}")
    print("=" * 90)
    for s in sessions:
        task_snippet = s.get("task", "")[:24]
        steps_info = f"{s.get('steps_completed',0)}/{s.get('steps_total',0)}"
        print(
            f"  {s.get('session_id',''):<34} "
            f"{s.get('status',''):<10} "
            f"{steps_info:<8} "
            f"{s.get('actions_total',0):<8} "
            f"{task_snippet}"
        )
    print("-" * 90)
    print("Use `python main.py --show-session <session_id>` or `--show-session latest` for details.\n")


def _handle_show_session(session_id: str) -> None:
    """Load and print full formatted session report."""
    target_id = session_id or "latest"
    session = load_session(target_id)
    if not session:
        print(f"Session not found: {target_id}", file=sys.stderr)
        sys.exit(1)
    print(format_session_summary(session))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computer Use Agent — Gemini plans, Jev executes"
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Natural language task to perform",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Generate the plan but don't execute it",
    )
    parser.add_argument(
        "--perceive",
        action="store_true",
        help="Just dump the current screen state and exit",
    )
    parser.add_argument(
        "--plan-file",
        type=str,
        default=None,
        help="Execute a plan from a JSON file instead of generating one",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Custom session identifier for logging",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List past execution sessions and exit",
    )
    parser.add_argument(
        "--show-session",
        nargs="?",
        const="latest",
        default=None,
        metavar="SESSION_ID",
        help="Display detailed summary of a session (defaults to latest)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable session logging to disk",
    )
    args = parser.parse_args()

    # ── Session Inspection Commands ────────────────────────────────────────
    if args.list_sessions:
        _handle_list_sessions()
        return

    if args.show_session is not None:
        _handle_show_session(args.show_session)
        return

    # ── Perceive-only mode ─────────────────────────────────────────────────
    if args.perceive:
        screen = perceive()
        print(json.dumps(screen.model_dump(), indent=2))
        return

    # ── Validate keys ──────────────────────────────────────────────────────
    if not config.GEMINI_API_KEY and not args.plan_file:
        print("Error: GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    if not config.TYPESAFE_API_KEY and not args.plan_only and not args.plan_file:
        print("Error: TYPESAFE_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # ── Initialize Session Logging ─────────────────────────────────────────
    task_desc = args.task or (
        f"Execute plan from {args.plan_file}" if args.plan_file else "Agent task"
    )
    session = None
    if not args.no_log:
        session = start_session(task=task_desc, session_id=args.session_id)
        print(f"[Session] Logging to {session.session_dir}")

    status = "success"
    err = None

    try:
        # ── Load or generate plan ──────────────────────────────────────────
        if args.plan_file:
            with open(args.plan_file) as f:
                plan = Plan.model_validate_json(f.read())
            print(f"Loaded plan from {args.plan_file}")
        elif args.task:
            print(f"Planning: {args.task}")
            plan = generate_plan(args.task)
            print(f"\nGenerated {len(plan.steps)} steps:")
            for step in plan.steps:
                print(
                    f"  {step.id}. [{step.action.value}] "
                    f"{step.target or step.text or step.key}"
                )
                print(f"     → {step.success_condition}")
        else:
            parser.print_help()
            sys.exit(1)

        # ── Plan-only mode ─────────────────────────────────────────────────
        if args.plan_only:
            print("\n--- Plan JSON ---")
            print(plan.model_dump_json(indent=2))
            return

        # ── Execute ────────────────────────────────────────────────────────
        execute_plan(plan)

    except Exception as e:
        status = "error"
        err = str(e)
        raise

    finally:
        if session:
            end_session(status=status, error=err)
            print(f"\n[Session Complete] View report: python main.py --show-session latest")


if __name__ == "__main__":
    main()

