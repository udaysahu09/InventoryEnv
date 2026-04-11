"""
InventoryEnv FastAPI Application
Complete OpenEnv compliant API for Meta Scaler Hackathon Round 1
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import logging

from environment import InventoryEnv

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS (Schemas)
# ============================================================================

class TaskType(str, Enum):
    """Enum for different task difficulties"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class InventoryObservation(BaseModel):
    """Observation schema for the InventoryEnv"""
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
    """Action schema for the InventoryEnv"""
    order_quantities: List[int] = Field(
        ...,
        description="Quantity to order for each product (non-negative integers)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "order_quantities": [20, 15, 10]
            }
        }


class RewardSchema(BaseModel):
    """Reward schema for the InventoryEnv"""
    reward: float = Field(
        ..., 
        description="Normalized reward strictly between 0.0 and 1.0 (exclusive)",
        gt=0.0, 
        lt=1.0
    )
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
    """Request schema for reset endpoint - MATCHES openenv.yaml"""
    task: str = Field(default="easy", description="Task difficulty level (easy, medium, hard)")

    def get_task(self) -> TaskType:
        """Extract and validate task from request"""
        try:
            return TaskType(self.task.lower() if isinstance(self.task, str) else "easy")
        except (ValueError, AttributeError):
            return TaskType.EASY


class StepRequest(BaseModel):
    """Request schema for step endpoint"""
    action: InventoryAction = Field(..., description="Action to take in the environment")


class StateResponse(BaseModel):
    """Response schema for state endpoint"""
    observation: InventoryObservation = Field(..., description="Current environment state")
    reward: Optional[RewardSchema] = Field(None, description="Last step reward")
    done: bool = Field(..., description="Whether episode is finished")


# ============================================================================
# GLOBAL STATE
# ============================================================================

env: InventoryEnv = None


# ============================================================================
# LIFESPAN CONTEXT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for FastAPI app initialization and cleanup"""
    global env
    try:
        logger.info("🚀 Starting InventoryEnv API")
        env = InventoryEnv(task=TaskType.EASY)
        logger.info("✅ Environment initialized with EASY task")
    except Exception as e:
        logger.error(f"❌ Failed to initialize environment: {str(e)}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down InventoryEnv API")


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="InventoryEnv API",
    description="OpenEnv API for B2B Supply Chain & Inventory Management - Meta Scaler Hackathon",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=True)
async def health_check():
    """
    Health check endpoint - Required for Hugging Face Spaces
    
    Returns:
        dict: API status information
    """
    return {
        "status": "ok",
        "message": "InventoryEnv API is running",
        "version": "1.0.0"
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Reset the environment to initial state
    
    **OpenEnv Standard Endpoint**
    
    Args:
        request (ResetRequest): Contains task difficulty level (easy, medium, hard)
        
    Returns:
        dict: Contains:
            - observation: Initial environment observation
            - message: Confirmation message
        
    Raises:
        HTTPException: 500 if reset fails
    """
    global env
    try:
        task = request.get_task()
        logger.info(f"📦 Resetting environment with task: {task.value}")
        
        # Create new environment instance
        env = InventoryEnv(task=task)
        observation = env.reset()
        
        logger.info(f"✅ Reset successful for task: {task.value}")
        
        return {
            "observation": observation.model_dump(),
            "message": f"Environment reset with task: {task.value}",
        }
    except Exception as e:
        logger.error(f"❌ Error during reset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {str(e)}"
        )


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute one step in the environment
    
    **OpenEnv Standard Endpoint**
    
    Args:
        request (StepRequest): Contains action with order quantities
        
    Returns:
        dict: Contains:
            - observation: Current environment state after action
            - reward: Reward breakdown with detailed metrics
            - done: Whether episode is complete
        
    Raises:
        HTTPException: 
            - 400 if environment not initialized (call /reset first)
            - 500 if step execution fails
    """
    global env
    
    # Check if environment is initialized
    if env is None:
        logger.warning("⚠️ Step called without environment initialization")
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        action_quantities = request.action.order_quantities
        logger.debug(f"📤 Executing step with action: {action_quantities}")
        
        # Execute step in environment
        observation, reward, done = env.step(request.action)
        
        logger.debug(f"✅ Step complete - Reward: {reward.reward:.4f}, Done: {done}")
        
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
        }
    except Exception as e:
        logger.error(f"❌ Error during step: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Step execution failed: {str(e)}"
        )


@app.get("/state")
async def state():
    """
    Get the current environment state
    
    **OpenEnv Standard Endpoint**
    
    Returns:
        dict: Contains:
            - observation: Current environment observation
            - reward: Last step's reward (None if no step executed yet)
            - done: Whether episode is complete
        
    Raises:
        HTTPException:
            - 400 if environment not initialized (call /reset first)
            - 500 if state retrieval fails
    """
    global env
    
    # Check if environment is initialized
    if env is None:
        logger.warning("⚠️ State requested without environment initialization")
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        logger.debug("📍 Retrieving environment state")
        state_data = env.state()
        return state_data
    except Exception as e:
        logger.error(f"❌ Error retrieving state: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"State retrieval failed: {str(e)}"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting InventoryEnv API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
