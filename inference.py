"""
InventoryEnv inference script for Meta Scaler Phase 2.
"""

import os
import sys
import time
from typing import Any, Dict, Optional

import requests


def _base_url() -> str:
    base_url = (
        os.environ.get("OPENENV_BASE_URL")
        or os.environ.get("API_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("CONTAINER_URL")
        or "http://localhost:7860"
    ).rstrip("/")
    
    print(f"[CONFIG] Using BASE_URL: {base_url}", file=sys.stderr)
    return base_url


BASE_URL = _base_url()


def wait_for_server(timeout_s: int = 60) -> None:
    """Wait until GET / returns HTTP 200 with retries."""
    deadline = time.time() + timeout_s
    attempt = 0
    last_err: Optional[Exception] = None

    while time.time() < deadline:
        attempt += 1
        try:
            print(f"[ATTEMPT {attempt}] Checking {BASE_URL}/", file=sys.stderr)
            r = requests.get(f"{BASE_URL}/", timeout=5)
            print(f"[RESPONSE] Status: {r.status_code}", file=sys.stderr)
            if r.status_code == 200:
                print(f"[SUCCESS] Server ready", file=sys.stderr)
                return
        except Exception as e:
            last_err = e
            print(f"[ATTEMPT {attempt} FAILED] {type(e).__name__}: {e}", file=sys.stderr)
        
        time.sleep(2)

    raise RuntimeError(
        f"Server {BASE_URL} not ready after {timeout_s}s. "
        f"Last error: {last_err}"
    )


def post_json(
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
    retries: int = 2
) -> Dict[str, Any]:
    """POST with retry logic."""
    url = f"{BASE_URL}{path}"
    last_err = None
    
    for attempt in range(retries + 1):
        try:
            print(f"[POST {attempt+1}/{retries+1}] {url}", file=sys.stderr)
            print(f"[PAYLOAD] {payload}", file=sys.stderr)
            
            r = requests.post(url, json=payload, timeout=timeout)
            
            print(f"[RESPONSE] Status: {r.status_code}", file=sys.stderr)
            if r.status_code >= 400:
                print(f"[ERROR] Response: {r.text}", file=sys.stderr)
            
            r.raise_for_status()
            result = r.json()
            print(f"[SUCCESS] Got response with keys: {list(result.keys())}", file=sys.stderr)
            return result
            
        except requests.exceptions.Timeout as e:
            last_err = e
            print(f"[ATTEMPT {attempt+1}] Timeout: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(2)
        except requests.exceptions.ConnectionError as e:
            last_err = e
            print(f"[ATTEMPT {attempt+1}] Connection error: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(2)
        except requests.exceptions.HTTPError as e:
            print(f"[HTTP ERROR {e.response.status_code}] {e.response.text}", file=sys.stderr)
            raise
        except Exception as e:
            last_err = e
            print(f"[ATTEMPT {attempt+1}] {type(e).__name__}: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(2)
    
    raise RuntimeError(f"POST {url} failed after {retries+1} attempts. Last error: {last_err}")


def get_json(path: str, timeout: int = 15, retries: int = 2) -> Dict[str, Any]:
    """GET with retry logic."""
    url = f"{BASE_URL}{path}"
    last_err = None
    
    for attempt in range(retries + 1):
        try:
            print(f"[GET {attempt+1}/{retries+1}] {url}", file=sys.stderr)
            r = requests.get(url, timeout=timeout)
            
            print(f"[RESPONSE] Status: {r.status_code}", file=sys.stderr)
            if r.status_code >= 400:
                print(f"[ERROR] Response: {r.text}", file=sys.stderr)
            
            r.raise_for_status()
            result = r.json()
            print(f"[SUCCESS] Got response with keys: {list(result.keys())}", file=sys.stderr)
            return result
            
        except requests.exceptions.Timeout as e:
            last_err = e
            print(f"[ATTEMPT {attempt+1}] Timeout: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(2)
        except requests.exceptions.HTTPError as e:
            print(f"[HTTP ERROR {e.response.status_code}] {e.response.text}", file=sys.stderr)
            raise
        except Exception as e:
            last_err = e
            print(f"[ATTEMPT {attempt+1}] {type(e).__name__}: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(2)
    
    raise RuntimeError(f"GET {url} failed after {retries+1} attempts. Last error: {last_err}")


def smoke_test() -> None:
    """Smoke test the environment."""
    wait_for_server(timeout_s=60)

    print("[TEST] === RESET ===", file=sys.stderr)
    reset_data = post_json("/reset", {"task": "easy"}, timeout=15, retries=2)
    
    obs = reset_data.get("observation")
    if not isinstance(obs, dict):
        raise RuntimeError(f"Invalid /reset response: {reset_data}")
    
    print(f"[TEST] Got observation with keys: {list(obs.keys())}", file=sys.stderr)

    print("[TEST] === STEP ===", file=sys.stderr)
    step_data = post_json(
        "/step",
        {"action": {"order_quantities": [0]}},
        timeout=15,
        retries=2
    )
    
    if "observation" not in step_data or "reward" not in step_data:
        raise RuntimeError(f"Invalid /step response: {step_data}")
    
    print(f"[TEST] Step succeeded", file=sys.stderr)

    print("[TEST] === STATE ===", file=sys.stderr)
    try:
        state_data = get_json("/state", timeout=15, retries=2)
        print(f"[TEST] Got state with keys: {list(state_data.keys())}", file=sys.stderr)
    except Exception as e:
        print(f"[WARNING] /state call failed (non-fatal): {e}", file=sys.stderr)


def main() -> int:
    try:
        print("[START] InventoryEnv inference validation", file=sys.stderr)
        smoke_test()
        print("[SUCCESS] All tests passed!", file=sys.stderr)
        print("inference.py OK")
        return 0
    except Exception as e:
        print(f"[FAILED] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
