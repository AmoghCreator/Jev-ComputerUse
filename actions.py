"""Actions — low-level desktop interaction via ydotool and hyprctl.

Every function here does ONE thing and is fully deterministic.
"""

from __future__ import annotations

from datetime import datetime
import subprocess
import time

import config
from models import ActionRecord


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=10, check=check
    )


def _settle() -> None:
    """Wait for the screen to settle after an action."""
    time.sleep(config.ACTION_SETTLE_DELAY_MS / 1000)


# ── Mouse ──────────────────────────────────────────────────────────────────────

def mouse_move(x: int, y: int) -> ActionRecord:
    """Move cursor to absolute screen coordinates."""
    t0 = time.time()
    res = _run(["ydotool", "mousemove", "--absolute", "-x", str(x), "-y", str(y)])
    _settle()
    duration_ms = (time.time() - t0) * 1000
    status = "success" if res.returncode == 0 else "failed"
    return ActionRecord(
        action="mouse_move",
        params={"x": x, "y": y},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res.stderr.strip() if status == "failed" else None,
    )


def mouse_click(x: int, y: int, button: str = "left") -> ActionRecord:
    """Move cursor to (x,y) and click.

    button: 'left', 'right', 'middle'
    """
    t0 = time.time()
    mouse_move(x, y)
    # ydotool click: 0xC0 = left press+release, 0xC1 = right, 0xC2 = middle
    btn_code = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}.get(
        button, "0xC0"
    )
    res = _run(["ydotool", "click", btn_code])
    _settle()
    duration_ms = (time.time() - t0) * 1000
    status = "success" if res.returncode == 0 else "failed"
    return ActionRecord(
        action="mouse_click",
        params={"x": x, "y": y, "button": button},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res.stderr.strip() if status == "failed" else None,
    )


def mouse_double_click(x: int, y: int) -> ActionRecord:
    """Double-click at (x, y)."""
    t0 = time.time()
    mouse_move(x, y)
    res1 = _run(["ydotool", "click", "0xC0"])
    time.sleep(0.05)
    res2 = _run(["ydotool", "click", "0xC0"])
    _settle()
    duration_ms = (time.time() - t0) * 1000
    status = "success" if (res1.returncode == 0 and res2.returncode == 0) else "failed"
    return ActionRecord(
        action="mouse_double_click",
        params={"x": x, "y": y},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res2.stderr.strip() if status == "failed" else None,
    )



# ── Keyboard ───────────────────────────────────────────────────────────────────

# Map friendly key names → ydotool keycodes (linux input event codes)
_KEY_MAP: dict[str, str] = {
    "return": "28",
    "enter": "28",
    "escape": "1",
    "esc": "1",
    "tab": "15",
    "space": "57",
    "backspace": "14",
    "delete": "111",
    "up": "103",
    "down": "108",
    "left": "105",
    "right": "106",
    "home": "102",
    "end": "107",
    "page_up": "104",
    "page_down": "109",
    "f1": "59",
    "f2": "60",
    "f3": "61",
    "f4": "62",
    "f5": "63",
    "f6": "64",
    "f7": "65",
    "f8": "66",
    "f9": "67",
    "f10": "68",
    "f11": "87",
    "f12": "88",
    "ctrl": "29",
    "alt": "56",
    "shift": "42",
    "super": "125",
    "a": "30",
    "b": "48",
    "c": "46",
    "d": "32",
    "e": "18",
    "f": "33",
    "g": "34",
    "h": "35",
    "i": "23",
    "j": "36",
    "k": "37",
    "l": "38",
    "m": "50",
    "n": "49",
    "o": "24",
    "p": "25",
    "q": "16",
    "r": "19",
    "s": "31",
    "t": "20",
    "u": "22",
    "v": "47",
    "w": "17",
    "x": "45",
    "y": "21",
    "z": "44",
}


def type_text(text: str) -> ActionRecord:
    """Type a string of text using ydotool."""
    t0 = time.time()
    res = _run(
        [
            "ydotool",
            "type",
            "--key-delay",
            str(config.YDOTOOL_TYPE_DELAY_MS),
            "--",
            text,
        ]
    )
    _settle()
    duration_ms = (time.time() - t0) * 1000
    status = "success" if res.returncode == 0 else "failed"
    return ActionRecord(
        action="type_text",
        params={"text": text},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res.stderr.strip() if status == "failed" else None,
    )


def press_key(key_spec: str) -> ActionRecord:
    """Press a key or key combo.

    key_spec examples:
      "Return"
      "ctrl+a"
      "ctrl+shift+t"
      "super"
    """
    t0 = time.time()
    parts = [p.strip().lower() for p in key_spec.split("+")]
    codes = []
    unknown = []
    for p in parts:
        code = _KEY_MAP.get(p)
        if code is None:
            unknown.append(p)
        else:
            codes.append(code)

    if unknown:
        msg = f"Unknown keys: {unknown}"
        print(f"[actions] {msg}, skipping")
        return ActionRecord(
            action="press_key",
            params={"key_spec": key_spec},
            timestamp=datetime.now().isoformat(),
            duration_ms=0.0,
            status="failed",
            details=msg,
        )

    # Build ydotool key sequence: press all, then release in reverse
    # Format: "code:1" = press, "code:0" = release
    seq = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
    res = _run(["ydotool", "key", *seq])
    _settle()
    duration_ms = (time.time() - t0) * 1000
    status = "success" if res.returncode == 0 else "failed"
    return ActionRecord(
        action="press_key",
        params={"key_spec": key_spec},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res.stderr.strip() if status == "failed" else None,
    )


# ── Window management (hyprctl) ────────────────────────────────────────────────

def launch_app(app_name: str) -> ActionRecord:
    """Launch an application via hyprctl dispatch exec, then focus it."""
    t0 = time.time()
    res = _run(["hyprctl", "dispatch", "exec", app_name])
    # Wait for the app to start and its window to appear
    time.sleep(2.0)
    import json

    focused_addr = None
    raw = _run(["hyprctl", "clients", "-j"])
    if raw.stdout:
        try:
            clients = json.loads(raw.stdout)
            for c in reversed(clients):  # newest windows tend to be last
                cls = c.get("class", "").lower()
                title = c.get("title", "").lower()
                if app_name.lower() in cls or app_name.lower() in title:
                    addr = c.get("address", "")
                    if addr:
                        _run(["hyprctl", "dispatch", "focuswindow", f"address:{addr}"])
                        time.sleep(0.5)
                        focused_addr = addr
                        break
        except Exception:
            pass

    duration_ms = (time.time() - t0) * 1000
    status = "success" if res.returncode == 0 else "failed"
    return ActionRecord(
        action="launch_app",
        params={"app_name": app_name, "focused_address": focused_addr},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res.stderr.strip() if status == "failed" else None,
    )


def focus_window(window_spec: str) -> ActionRecord:
    """Focus a window by class, title, or address."""
    t0 = time.time()
    import json

    addr = None
    raw = _run(["hyprctl", "clients", "-j"])
    if raw.stdout:
        try:
            clients = json.loads(raw.stdout)
            spec_lower = window_spec.lower().strip()
            for c in reversed(clients):
                c_cls = c.get("class", "").lower()
                c_title = c.get("title", "").lower()
                c_addr = c.get("address", "").lower()
                if (
                    spec_lower == c_addr
                    or spec_lower in c_cls
                    or spec_lower in c_title
                ):
                    addr = c.get("address")
                    break
        except Exception:
            pass

    if addr:
        res = _run(["hyprctl", "dispatch", "focuswindow", f"address:{addr}"])
    else:
        res = _run(["hyprctl", "dispatch", "focuswindow", f"class:{window_spec}"])

    _settle()
    duration_ms = (time.time() - t0) * 1000
    status = "success" if res.returncode == 0 else "failed"
    return ActionRecord(
        action="focus_window",
        params={"window_spec": window_spec, "address": addr},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res.stderr.strip() if status == "failed" else None,
    )



def scroll(direction: str = "down", amount: int = 3) -> ActionRecord:
    """Scroll the mouse wheel.

    direction: 'up' or 'down'
    """
    t0 = time.time()
    # ydotool mousemove --wheel: negative = up, positive = down
    delta = amount if direction == "down" else -amount
    res = _run(["ydotool", "mousemove", "--wheel", str(delta)])
    _settle()
    duration_ms = (time.time() - t0) * 1000
    status = "success" if res.returncode == 0 else "failed"
    return ActionRecord(
        action="scroll",
        params={"direction": direction, "amount": amount},
        timestamp=datetime.now().isoformat(),
        duration_ms=round(duration_ms, 2),
        status=status,
        details=res.stderr.strip() if status == "failed" else None,
    )

