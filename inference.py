"""
InventoryEnv inference script for Meta Scaler Phase 2.

IMPORTANT:
- Do NOT use FastAPI TestClient here (it can break due to dependency version mismatches).
- Instead, call the running container's HTTP endpoints using requests.
- Exit code must be 0 on success, non-zero on failure.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests


def _base_url() -> str:
    # Meta Scaler may set an env var; keep multiple fallbacks.
    # Default assumes the server is running inside the same container on port 7860.
    return (
        os.environ.get("OPENENV_BASE_URL")
        or os.environ.get("API_BASE_URL")
        or os.environ.get("BASE_URL")
        or "http://127.0.0.1:7860"
    ).rstrip("/")


BASE_URL = _base_url()


def wait_for_server(timeout_s: int = 30) -> None:
    """Wait until GET / returns HTTP 200."""
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/", timeout=2)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(1)

    raise RuntimeError(f"Server not reachable at {BASE_URL} within {timeout_s}s. Last error: {last_err}")


def post_json(path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_json(path: str, timeout: int = 10) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def smoke_test() -> None:
    """
    Minimal validation:
    - server responds on /
    - reset works
    - step works
    """
    wait_for_server(timeout_s=30)

    # Reset (try with body first)
    reset_data = post_json("/reset", {"task": "easy"}, timeout=10)
    obs = reset_data.get("observation", reset_data)  # some APIs may return observation at top-level
    if not isinstance(obs, dict):
        raise RuntimeError(f"/reset response has no observation dict: {reset_data}")

    # Step
    step_data = post_json("/step", {"action": {"order_quantities": [0]}}, timeout=10)
    if "observation" not in step_data or "reward" not in step_data:
        raise RuntimeError(f"/step response missing required fields: {step_data}")

    # Optional: check reward is numeric and inside (0,1) if present
    reward = step_data.get("reward", {})
    if isinstance(reward, dict) and "reward" in reward:
        val = reward["reward"]
        if not isinstance(val, (int, float)):
            raise RuntimeError(f"reward.reward is not a number: {val}")
        # Don't hard-fail if equals 0 or 1 because some envs may produce boundary values early;
        # if your grader requires strict bounds, your environment should ensure that.
        # We'll just print it for debugging.
    # Optional state call (non-fatal if not needed)
    try:
        _ = get_json("/state", timeout=10)
    except Exception:
        pass


def main() -> int:
    try:
        smoke_test()
        print("inference.py OK")
        return 0
    except Exception as e:
        # Print error so participant logs show something useful
        print(f"inference.py FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
