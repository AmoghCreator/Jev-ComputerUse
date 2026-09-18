"""Configuration — API keys, timeouts, retry limits."""

import os

from dotenv import load_dotenv

load_dotenv()  # loads .env file if present


# API Keys (read from environment)
TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Jev model
JEV_MODEL = "jev-latest"

# Gemini model
GEMINI_MODEL = "gemini-3.1-pro-preview"

# Execution loop
MAX_RETRIES_PER_STEP = 10
STEP_POLL_DELAY_MS = 500  # ms to wait between perception cycles
ACTION_SETTLE_DELAY_MS = 300  # ms to wait after executing an action

# ydotool
YDOTOOL_SOCKET = os.environ.get(
    "YDOTOOL_SOCKET", "/run/user/1000/.ydotool_socket"
)
YDOTOOL_KEY_DELAY_MS = 20
YDOTOOL_TYPE_DELAY_MS = 20

# Logging
LOGS_DIR = os.environ.get(
    "LOGS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
)
ENABLE_SESSION_LOGGING = os.environ.get("ENABLE_SESSION_LOGGING", "true").lower() in (
    "true",
    "1",
    "yes",
)

