"""Perceiver — converts the live desktop into a structured ScreenState.

Combines two sources:
  1. hyprctl  → window list, active window, cursor position
  2. AT-SPI   → interactive UI elements inside the focused window
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional

from models import ScreenState, UIElement, WindowInfo


# ── hyprctl helpers ────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    return result.stdout.strip()


def _get_cursor_pos() -> tuple[int, int]:
    raw = _run(["hyprctl", "cursorpos"])
    # Output: "1418, 402"
    parts = raw.replace(",", "").split()
    return int(parts[0]), int(parts[1])


def _get_active_window_address() -> Optional[str]:
    raw = _run(["hyprctl", "activewindow", "-j"])
    if not raw:
        return None
    data = json.loads(raw)
    return data.get("address")


def _get_windows() -> list[WindowInfo]:
    raw = _run(["hyprctl", "clients", "-j"])
    if not raw:
        return []
    clients = json.loads(raw)
    active_addr = _get_active_window_address()
    windows: list[WindowInfo] = []
    for c in clients:
        if not c.get("mapped", False):
            continue
        at = c.get("at", [0, 0])
        size = c.get("size", [0, 0])
        ws = c.get("workspace", {})
        windows.append(
            WindowInfo(
                **{
                    "class": c.get("class", ""),
                    "title": c.get("title", ""),
                    "x": at[0],
                    "y": at[1],
                    "width": size[0],
                    "height": size[1],
                    "workspace": str(ws.get("name", "")),
                    "focused": c.get("address") == active_addr,
                }
            )
        )
    return windows


# ── AT-SPI helpers ─────────────────────────────────────────────────────────────

# Roles we care about for interaction
_INTERACTIVE_ROLES = {
    "push button",
    "toggle button",
    "radio button",
    "check box",
    "menu item",
    "menu",
    "text",
    "entry",
    "password text",
    "combo box",
    "list item",
    "link",
    "tab",
    "page tab",
    "slider",
    "spin button",
    "tool bar item",
    "tree item",
}


def _get_atspi_tree(active_window: WindowInfo | None) -> list[UIElement]:
    """Walk the AT-SPI tree and extract a hierarchical tree of interactive elements."""
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        desktop = Atspi.get_desktop(0)
        
        win_x = active_window.x if active_window else 0
        win_y = active_window.y if active_window else 0

        def walk(node: object, path: str, depth: int = 0) -> list[UIElement]:
            if depth > 15:  # safety limit
                return []
                
            children_elements = []
            try:
                for i in range(node.get_child_count()):
                    child = node.get_child_at_index(i)
                    if child:
                        child_path = f"{path}.{i}" if path else str(i)
                        children_elements.extend(walk(child, child_path, depth + 1))
            except Exception:
                pass

            try:
                role = node.get_role_name()
                name = node.get_name() or ""

                if role in _INTERACTIVE_ROLES and name:
                    comp = node.get_component_iface()
                    if comp:
                        ext = comp.get_extents(0)  # screen coords
                        if ext.width > 0 and ext.height > 0:
                            el = UIElement(
                                role=role.replace(" ", "_"),
                                name=name,
                                x=ext.x,
                                y=ext.y,
                                w=ext.width,
                                h=ext.height,
                                path=path,
                                relative_x=ext.x - win_x,
                                relative_y=ext.y - win_y,
                                children=children_elements
                            )
                            return [el]
            except Exception:
                pass

            # If node is not interactive itself but has interactive children, flatten them up.
            return children_elements
            
        tree = []
        for i in range(desktop.get_child_count()):
            app = desktop.get_child_at_index(i)
            if app and app.get_child_count() > 0:
                tree.extend(walk(app, str(i)))
                
        return tree

    except Exception:
        # AT-SPI not available or failed — return empty
        return []


# ── Public API ─────────────────────────────────────────────────────────────────

def perceive() -> ScreenState:
    """Capture the current screen state as a structured object."""
    cx, cy = _get_cursor_pos()
    windows = _get_windows()
    active = next((w for w in windows if w.focused), None)
    ui_tree = _get_atspi_tree(active)

    return ScreenState(
        cursor_x=cx,
        cursor_y=cy,
        active_window=active,
        windows=windows,
        ui_tree=ui_tree,
    )
