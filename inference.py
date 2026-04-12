"""
InventoryEnv inference script for Meta Scaler Phase 2.
"""

import os
import sys
import time
from typing import Any, Dict, Optional
import logging

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def get_base_url() -> str:
    """Determine the correct base URL"""
    
    # Get from environment
    base_url = (
        os.environ.get("OPENENV_BASE_URL")
        or os.environ.get("API_BASE_URL")
        or os.environ.get("BASE_URL")
        or "http://localhost:7860"
    ).rstrip("/")
    
    logger.info(f"[CONFIG] Base URL from env: {base_url}")
    return base_url


BASE_URL = get_base_url()


def wait_for_server(timeout_s: int = 60) -> bool:
    """Wait for server to be ready"""
    deadline = time.time() + timeout_s
    
    while time.time() < deadline:
        try:
            logger.info(f"[HEALTH CHECK] GET {BASE_URL}/")
            r = requests.get(f"{BASE_URL}/", timeout=5)
            if r.status_code == 200:
                logger.info(f"[SUCCESS] Server is ready")
                return True
        except Exception as e:
            logger.info(f"[RETRY] {type(e).__name__}: {e}")
        
        time.sleep(2)
    
    logger.error(f"[FAILED] Server not ready after {timeout_s}s")
    return False


def smoke_test() -> None:
    """Run smoke tests on all required endpoints"""
    
    # Step 1: Wait for server
    if not wait_for_server(timeout_s=60):
        raise RuntimeError("Server initialization timeout")
    
    # Step 2: Test /reset
    logger.info(f"[TEST 1/3] POST /reset")
    try:
        r = requests.post(
            f"{BASE_URL}/reset",
            json={"task": "easy"},
            timeout=15
        )
        logger.info(f"[RESPONSE] HTTP {r.status_code}")
        
        if r.status_code >= 400:
            logger.error(f"[ERROR] {r.text}")
            raise RuntimeError(f"POST /reset failed: HTTP {r.status_code}")
        
        reset_data = r.json()
        if "observation" not in reset_data:
            raise RuntimeError(f"Invalid response: missing 'observation'")
        
        logger.info(f"[SUCCESS] /reset returned observation")
        
    except Exception as e:
        logger.error(f"[FAILED] /reset test: {e}")
        raise
    
    # Step 3: Test /step
    logger.info(f"[TEST 2/3] POST /step")
    try:
        r = requests.post(
            f"{BASE_URL}/step",
            json={"action": {"order_quantities": [0]}},
            timeout=15
        )
        logger.info(f"[RESPONSE] HTTP {r.status_code}")
        
        if r.status_code >= 400:
            logger.error(f"[ERROR] {r.text}")
            raise RuntimeError(f"POST /step failed: HTTP {r.status_code}")
        
        step_data = r.json()
        if "observation" not in step_data or "reward" not in step_data:
            raise RuntimeError(f"Invalid response: missing required fields")
        
        logger.info(f"[SUCCESS] /step returned observation and reward")
        
    except Exception as e:
        logger.error(f"[FAILED] /step test: {e}")
        raise
    
    # Step 4: Test /state
    logger.info(f"[TEST 3/3] GET /state")
    try:
        r = requests.get(
            f"{BASE_URL}/state",
            timeout=15
        )
        logger.info(f"[RESPONSE] HTTP {r.status_code}")
        
        if r.status_code >= 400:
            logger.error(f"[ERROR] {r.text}")
            # /state is optional, don't fail
            logger.warning(f"[WARNING] /state not available (non-fatal)")
        else:
            state_data = r.json()
            logger.info(f"[SUCCESS] /state returned data")
        
    except Exception as e:
        logger.warning(f"[WARNING] /state test failed: {e} (non-fatal)")


def main() -> int:
    """Main entry point"""
    try:
        logger.info(f"[START] InventoryEnv inference validation")
        logger.info(f"[CONFIG] Connecting to: {BASE_URL}")
        
        smoke_test()
        
        logger.info(f"[SUCCESS] All tests passed!")
        print("inference.py OK")
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
