"""
InventoryEnv FastAPI Application
Complete OpenEnv compliant API for Meta Scaler Hackathon Round 1
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional

from models import TaskType, InventoryObservation, InventoryAction, RewardSchema
from environment import InventoryEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResetRequest(BaseModel):
    task: str = Field(default="easy", description="Task difficulty level (easy, medium, hard)")

    def get_task(self) -> TaskType:
        try:
            return TaskType(self.task.lower() if isinstance(self.task, str) else "easy")
        except (ValueError, AttributeError):
            return TaskType.EASY


class StepRequest(BaseModel):
    action: InventoryAction = Field(..., description="Action to take in the environment")


class StateResponse(BaseModel):
    observation: InventoryObservation = Field(..., description="Current environment state")
    reward: Optional[RewardSchema] = Field(None, description="Last step reward")
    done: bool = Field(..., description="Whether episode is finished")


env: InventoryEnv = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    try:
        logger.info("Starting InventoryEnv API")
        env = InventoryEnv(task=TaskType.EASY)
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise
    yield
    logger.info("Shutting down InventoryEnv API")


# Get root_path from environment variable (set by Meta Scaler proxy)
root_path = os.environ.get("ROOT_PATH", "").rstrip("/")
logger.info(f"[CONFIG] ROOT_PATH: {root_path if root_path else '(none)'}")

app = FastAPI(
    title="InventoryEnv API",
    description="OpenEnv API for B2B Supply Chain & Inventory Management",
    version="1.0.0",
    lifespan=lifespan,
    root_path=root_path,  # CRITICAL: Tell FastAPI about proxy prefix
)


@app.get("/")
async def health_check():
    """Health check endpoint"""
    logger.info("[ENDPOINT] GET /")
    return {
        "status": "ok",
        "message": "InventoryEnv API is running",
        "version": "1.0.0"
    }


@app.post("/reset")
async def reset(request: ResetRequest = Body(default=ResetRequest())):
    """Reset the environment to initial state"""
    global env
    logger.info(f"[ENDPOINT] POST /reset - task={request.task}")
    try:
        task = request.get_task()
        logger.info(f"Resetting environment with task: {task.value}")
        env = InventoryEnv(task=task)
        observation = env.reset()
        
        return {
            "observation": observation.model_dump(),
            "message": f"Environment reset with task: {task.value}",
        }
    except Exception as e:
        logger.error(f"Error during reset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """Execute one step in the environment"""
    global env
    logger.info(f"[ENDPOINT] POST /step")
    if env is None:
        logger.error("Environment not initialized")
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        observation, reward, done = env.step(request.action)
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
        }
    except Exception as e:
        logger.error(f"Error during step: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Get the current environment state"""
    global env
    logger.info(f"[ENDPOINT] GET /state")
    if env is None:
        logger.error("Environment not initialized")
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        state_data = env.state()
        return state_data
    except Exception as e:
        logger.error(f"Error retrieving state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point for the application"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
