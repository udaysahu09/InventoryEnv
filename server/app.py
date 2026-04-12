"""
InventoryEnv FastAPI Application
Complete OpenEnv compliant API for Meta Scaler Hackathon Round 1
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional

from models import TaskType, InventoryObservation, InventoryAction, RewardSchema
from environment import InventoryEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
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
    """Application lifecycle manager"""
    global env
    try:
        logger.info("=" * 60)
        logger.info("Starting InventoryEnv API")
        logger.info("=" * 60)
        env = InventoryEnv(task=TaskType.EASY)
        logger.info("Environment initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize environment: {str(e)}", exc_info=True)
        raise
    yield
    logger.info("Shutting down InventoryEnv API")


# Create FastAPI app with proxy support
app = FastAPI(
    title="InventoryEnv API",
    description="OpenEnv API for B2B Supply Chain & Inventory Management",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for proxy compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    """Health check endpoint"""
    logger.info("GET / - Health check")
    return {
        "status": "ok",
        "message": "InventoryEnv API is running",
        "version": "1.0.0",
        "healthy": True
    }


@app.get("/health")
async def health_status():
    """Alternative health check endpoint"""
    logger.info("GET /health - Health status")
    return {
        "status": "ok",
        "healthy": env is not None
    }


@app.post("/reset")
async def reset(request: ResetRequest = Body(default=ResetRequest())):
    """Reset the environment to initial state
    
    Request body:
    {
        "task": "easy|medium|hard"
    }
    """
    global env
    logger.info(f"POST /reset - Starting reset with task={request.task}")
    
    try:
        task = request.get_task()
        logger.info(f"Task parsed as: {task.value}")
        
        # Reinitialize environment
        env = InventoryEnv(task=task)
        observation = env.reset()
        
        logger.info(f"Environment reset successfully")
        
        return {
            "observation": observation.model_dump(),
            "message": f"Environment reset with task: {task.value}",
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Error during reset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(request: StepRequest):
    """Execute one step in the environment
    
    Request body:
    {
        "action": {
            "order_quantities": [int, int, int]
        }
    }
    """
    global env
    logger.info(f"POST /step - Starting step")
    
    if env is None:
        logger.warning("Environment not initialized - returning 400")
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        logger.info(f"Action: {request.action}")
        observation, reward, done = env.step(request.action)
        
        logger.info(f"Step completed - done={done}")
        
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Error during step: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def state():
    """Get the current environment state"""
    global env
    logger.info(f"GET /state - Fetching state")
    
    if env is None:
        logger.warning("Environment not initialized - returning 400")
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        state_data = env.state()
        logger.info(f"State retrieved successfully")
        return state_data
    except Exception as e:
        logger.error(f"Error retrieving state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


def main():
    """Main entry point for the application"""
    import uvicorn
    logger.info(f"Starting Uvicorn server on 0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
