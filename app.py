"""
InventoryEnv FastAPI Application
Complete OpenEnv compliant API for Meta Scaler Hackathon Round 1
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import logging

from environment import InventoryEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class InventoryObservation(BaseModel):
    current_day: int = Field(..., description="Current simulation day (0 to max_days)")
    stock_levels: List[int] = Field(..., description="Current stock level for each product")
    warehouse_capacity: int = Field(..., description="Total warehouse capacity")
    warehouse_used: int = Field(..., description="Current warehouse space used")
    pending_orders: List[int] = Field(..., description="Pending orders for each product")
    total_fulfilled: int = Field(..., description="Total orders fulfilled so far")
    total_demand: int = Field(..., description="Total orders generated so far")
    current_balance: float = Field(..., description="Current account balance (revenue - costs)")
    task: str = Field(..., description="Current task difficulty level")

    class Config:
        json_schema_extra = {
            "example": {
                "current_day": 5,
                "stock_levels": [50, 30, 20],
                "warehouse_capacity": 200,
                "warehouse_used": 100,
                "pending_orders": [10, 5, 8],
                "total_fulfilled": 45,
                "total_demand": 60,
                "current_balance": 1000.0,
                "task": "medium"
            }
        }

class InventoryAction(BaseModel):
    order_quantities: List[int] = Field(..., description="Quantity to order for each product")

    class Config:
        json_schema_extra = {
            "example": {
                "order_quantities": [20, 15, 10]
            }
        }

class RewardSchema(BaseModel):
    reward: float = Field(..., description="Normalized reward strictly between 0.0 and 1.0 (exclusive)", gt=0.0, lt=1.0)
    fulfilled_percentage: float = Field(..., description="Percentage of orders fulfilled (0.0 to 100.0)")
    oos_penalty: float = Field(..., description="Out-of-stock penalty applied")
    storage_cost: float = Field(..., description="Storage cost for current inventory")
    order_cost: float = Field(..., description="Cost of orders placed")
    total_revenue: float = Field(..., description="Total revenue from fulfilled orders")
    done: bool = Field(..., description="Whether episode is complete")

    class Config:
        json_schema_extra = {
            "example": {
                "reward": 0.85,
                "fulfilled_percentage": 85.0,
                "oos_penalty": 0.05,
                "storage_cost": 50.0,
                "order_cost": 100.0,
                "total_revenue": 500.0,
                "done": False
            }
        }

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

app = FastAPI(
    title="InventoryEnv API",
    description="OpenEnv API for B2B Supply Chain & Inventory Management",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "message": "InventoryEnv API is running",
        "version": "1.0.0"
    }

@app.post("/reset")
async def reset(request: ResetRequest):
    global env
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
        logger.error(f"Error during reset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        observation, reward, done = env.step(request.action)
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
        }
    except Exception as e:
        logger.error(f"Error during step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        state_data = env.state()
        return state_data
    except Exception as e:
        logger.error(f"Error retrieving state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
