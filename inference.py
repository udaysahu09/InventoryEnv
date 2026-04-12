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
    # Try multiple environment variables that Meta Scaler might use
    base_url = (
        os.environ.get("OPENENV_BASE_URL")
        or os.environ.get("API_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("CONTAINER_URL")
        or os.environ.get("SERVER_URL")
        or "http://localhost:7860"  # Default for local testing
    ).rstrip("/")
    
    print(f"[DEBUG] Connecting to: {base_url}", file=sys.stderr)
    return base_url


BASE_URL = _base_url()


def wait_for_server(timeout_s: int = 30) -> None:
    """Wait until GET / returns HTTP 200."""
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/", timeout=2)
            if r.status_code == 200:
                print(f"[SUCCESS] Server responding at {BASE_URL}", file=sys.stderr)
                return
        except Exception as e:
            last_err = e
            print(f"[RETRY] Connection attempt failed: {e}", file=sys.stderr)
        time.sleep(1)

    raise RuntimeError(f"Server not reachable at {BASE_URL} within {timeout_s}s. Last error: {last_err}")


def post_json(path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] POST {url} failed with {e.response.status_code}: {e.response.text}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] POST {url} failed: {e}", file=sys.stderr)
        raise


def get_json(path: str, timeout: int = 10) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] GET {url} failed with {e.response.status_code}: {e.response.text}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] GET {url} failed: {e}", file=sys.stderr)
        raise


def smoke_test() -> None:
    """
    Minimal validation:
    - server responds on /
    - reset works
    - step works
    """
    wait_for_server(timeout_s=30)

    print("[TEST] Calling /reset with task=easy", file=sys.stderr)
    # Reset with proper payload
    reset_data = post_json("/reset", {"task": "easy"}, timeout=10)
    obs = reset_data.get("observation", reset_data)
    if not isinstance(obs, dict):
        raise RuntimeError(f"/reset response has no observation dict: {reset_data}")
    print("[TEST] /reset succeeded", file=sys.stderr)

    print("[TEST] Calling /step with order_quantities=[0]", file=sys.stderr)
    # Step
    step_data = post_json("/step", {"action": {"order_quantities": [0]}}, timeout=10)
    if "observation" not in step_data or "reward" not in step_data:
        raise RuntimeError(f"/step response missing required fields: {step_data}")
    print("[TEST] /step succeeded", file=sys.stderr)

    # Optional: check reward is numeric
    reward = step_data.get("reward", {})
    if isinstance(reward, dict) and "reward" in reward:
        val = reward["reward"]
        if not isinstance(val, (int, float)):
            raise RuntimeError(f"reward.reward is not a number: {val}")
        print(f"[TEST] reward value: {val}", file=sys.stderr)
    
    # Optional state call
    try:
        print("[TEST] Calling /state", file=sys.stderr)
        _ = get_json("/state", timeout=10)
        print("[TEST] /state succeeded", file=sys.stderr)
    except Exception:
        pass


def main() -> int:
    try:
        print(f"[START] InventoryEnv inference starting", file=sys.stderr)
        smoke_test()
        print("inference.py OK")
        return 0
    except Exception as e:
        print(f"inference.py FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
