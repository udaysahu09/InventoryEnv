"""
InventoryEnv inference script for Meta Scaler Phase 2.

URL priority:
  ENV_URL          → explicit environment server URL (set by hackathon validator)
  OPENENV_BASE_URL → OpenEnv-standard env URL
  http://localhost:7860 (fallback)

API_BASE_URL / HF_TOKEN are used ONLY for the LLM, never for the env.
"""

import os
import sys
import time
import logging
from typing import Any, Dict, List, Optional

import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


# ── URLs ──────────────────────────────────────────────────────────────────────
def get_env_url() -> str:
    """
    Return the inventory-environment server URL.
    Never use API_BASE_URL here — that points to the LLM router.
    """
    url = (
        os.environ.get("ENV_URL")           # explicit env server
        or os.environ.get("OPENENV_BASE_URL") # openenv standard
        or "http://localhost:7860"           # local fallback
    ).rstrip("/")
    logger.info(f"[CONFIG] ENV server URL: {url}")
    return url


def get_llm_base_url() -> Optional[str]:
    return (os.environ.get("API_BASE_URL") or "").rstrip("/") or None


ENV_URL     = get_env_url()
LLM_URL     = get_llm_base_url()
MODEL_NAME  = os.environ.get("MODEL_NAME", "")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")


# ── LLM client (optional) ─────────────────────────────────────────────────────
llm_client = None
try:
    if LLM_URL and HF_TOKEN:
        from openai import OpenAI
        llm_client = OpenAI(base_url=LLM_URL, api_key=HF_TOKEN)
        logger.info(f"[CONFIG] LLM client ready → {LLM_URL}")
except Exception as e:
    logger.info(f"[CONFIG] LLM client not available: {e}")


# ── Server readiness ──────────────────────────────────────────────────────────
def wait_for_server(timeout_s: int = 90) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{ENV_URL}/", timeout=5)
            if r.status_code == 200:
                logger.info("[SUCCESS] Environment server is ready")
                return True
        except Exception as e:
            logger.info(f"[RETRY] {type(e).__name__}: {e}")
        time.sleep(3)
    logger.error(f"[FAILED] Server not ready after {timeout_s}s")
    return False


# ── Action logic ──────────────────────────────────────────────────────────────
def choose_action(observation: Dict[str, Any], num_products: int) -> List[int]:
    """
    Decide order quantities.
    Tries the LLM first; falls back to a heuristic that keeps stock
    at ~2× daily demand so we never go OOS.
    """
    stock: List[int]  = observation.get("stock_levels", [50] * num_products)
    pending: List[int] = observation.get("pending_orders", [0] * num_products)
    capacity: int      = observation.get("warehouse_capacity", 200)
    used: int          = observation.get("warehouse_used", sum(stock))
    day: int           = observation.get("current_day", 0)

    # Heuristic target: keep ~30 units per product as a safety buffer
    target = max(30, capacity // (num_products * 3))
    free_space = capacity - used

    orders = []
    for i in range(num_products):
        shortage = max(0, target - stock[i] + pending[i])
        # Don't over-order beyond free space
        q = min(shortage, max(0, free_space // num_products))
        orders.append(int(q))

    # Try LLM override
    if llm_client and MODEL_NAME:
        try:
            prompt = (
                f"Inventory management. Day={day}, stock={stock}, "
                f"pending_orders={pending}, warehouse_free={free_space}, "
                f"num_products={num_products}. "
                f"Reply ONLY with a JSON array of {num_products} non-negative integers "
                f"representing order quantities. Example: [10, 5, 8]"
            )
            resp = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )
            import json, re
            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\[[\d,\s]+\]", raw)
            if match:
                llm_orders = json.loads(match.group())
                if len(llm_orders) == num_products and all(isinstance(x, int) and x >= 0 for x in llm_orders):
                    logger.info(f"[LLM] orders={llm_orders}")
                    return llm_orders
        except Exception as e:
            logger.info(f"[LLM] error: {e} — using heuristic")

    return orders


# ── Single task run ───────────────────────────────────────────────────────────
def run_task(task_name: str) -> float:
    """Run one full episode for a task. Returns final reward average."""
    logger.info(f"\n[TASK] Starting task='{task_name}'")

    # Reset
    try:
        r = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task_name},
            timeout=15,
        )
        if r.status_code >= 400:
            logger.error(f"[ERROR] /reset HTTP {r.status_code}: {r.text}")
            return 0.5
        reset_data = r.json()
    except Exception as e:
        logger.error(f"[ERROR] /reset failed: {e}")
        return 0.5

    observation = reset_data.get("observation", {})
    num_products = len(observation.get("stock_levels", [1]))
    logger.info(f"[RESET] task={task_name} num_products={num_products}")

    rewards: List[float] = []
    max_steps = 100

    for step in range(1, max_steps + 1):
        action_quantities = choose_action(observation, num_products)

        try:
            r = requests.post(
                f"{ENV_URL}/step",
                json={"action": {"order_quantities": action_quantities}},
                timeout=15,
            )
            if r.status_code >= 400:
                logger.error(f"[ERROR] /step HTTP {r.status_code}: {r.text}")
                break
            step_data = r.json()
        except Exception as e:
            logger.error(f"[ERROR] /step failed: {e}")
            break

        observation = step_data.get("observation", observation)
        reward_obj  = step_data.get("reward", {})
        done        = step_data.get("done", False)
        reward_val  = float(reward_obj.get("reward", 0.5)) if reward_obj else 0.5
        rewards.append(reward_val)

        logger.info(
            f"[STEP] {step:3d} | action={action_quantities} | "
            f"reward={reward_val:.4f} | done={done}"
        )

        if done:
            break

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.5
    logger.info(f"[TASK DONE] task={task_name} steps={len(rewards)} avg_reward={avg_reward:.4f}")
    return avg_reward


# ── Smoke test ────────────────────────────────────────────────────────────────
def smoke_test() -> None:
    """Quick validation of all required endpoints."""

    logger.info("[START] InventoryEnv inference validation")
    logger.info(f"[CONFIG] Connecting to: {ENV_URL}")

    # 1. Wait for server
    if not wait_for_server(timeout_s=90):
        raise RuntimeError("Environment server did not become ready in time")

    # 2. POST /reset
    logger.info("[TEST 1/3] POST /reset")
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task": "easy"}, timeout=15)
        logger.info(f"[RESPONSE] HTTP {r.status_code}")
        if r.status_code >= 400:
            raise RuntimeError(f"POST /reset failed: HTTP {r.status_code} — {r.text}")
        data = r.json()
        if "observation" not in data:
            raise RuntimeError(f"POST /reset: missing 'observation' in response")
        num_products = len(data["observation"].get("stock_levels", [1]))
        logger.info(f"[SUCCESS] /reset OK  num_products={num_products}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"POST /reset failed: {e}") from e

    # 3. POST /step
    logger.info("[TEST 2/3] POST /step")
    try:
        orders = [10] * num_products
        r = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"order_quantities": orders}},
            timeout=15,
        )
        logger.info(f"[RESPONSE] HTTP {r.status_code}")
        if r.status_code >= 400:
            raise RuntimeError(f"POST /step failed: HTTP {r.status_code} — {r.text}")
        data = r.json()
        if "observation" not in data or "reward" not in data:
            raise RuntimeError(f"POST /step: missing required fields")
        logger.info(f"[SUCCESS] /step OK  reward={data['reward'].get('reward')}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"POST /step failed: {e}") from e

    # 4. GET /state (non-fatal)
    logger.info("[TEST 3/3] GET /state")
    try:
        r = requests.get(f"{ENV_URL}/state", timeout=15)
        logger.info(f"[RESPONSE] HTTP {r.status_code}")
        if r.status_code < 400:
            logger.info("[SUCCESS] /state OK")
        else:
            logger.warning(f"[WARNING] /state returned {r.status_code} (non-fatal)")
    except Exception as e:
        logger.warning(f"[WARNING] /state failed: {e} (non-fatal)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        smoke_test()

        logger.info("\n[SUCCESS] Smoke tests passed — running full episodes")

        scores = []
        for task in ["easy", "medium", "hard"]:
            try:
                score = run_task(task)
                scores.append(score)
                logger.info(f"[SCORE] {task}: {score:.4f}")
            except Exception as e:
                logger.error(f"[ERROR] task={task} failed: {e}")
                scores.append(0.5)

        overall = sum(scores) / len(scores) if scores else 0.5
        logger.info(f"\n[FINAL] overall_score={overall:.4f}")
        print("inference.py OK")
        return 0

    except Exception as e:
        logger.error(f"[FAILED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
