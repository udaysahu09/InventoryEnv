from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from models import (
    InventoryObservation,
    InventoryAction,
    RewardSchema,
    ResetRequest,
    StepRequest,
    StateResponse,
    TaskType,
)
from environment import InventoryEnv

# Global environment instance
env: InventoryEnv = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for FastAPI app."""
    global env
    env = InventoryEnv(task=TaskType.EASY)
    yield
    # Cleanup if needed


app = FastAPI(
    title="InventoryEnv API",
    description="OpenEnv API for B2B Supply Chain & Inventory Management",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", status_code=200)
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "InventoryEnv API is running",
        "version": "1.0.0",
    }


@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Reset the environment.
    
    Parameters:
        - task: TaskType (easy, medium, hard)
    
    Returns:
        - observation: InventoryObservation
    """
    global env
    try:
        env = InventoryEnv(task=request.task)
        observation = env.reset()
        return {
            "observation": observation.model_dump(),
            "message": f"Environment reset with task: {request.task.value}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute one step in the environment.
    
    Parameters:
        - action: InventoryAction with order quantities
    
    Returns:
        - observation: InventoryObservation
        - reward: RewardSchema
        - done: bool
    """
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """
    Get the current environment state.
    
    Returns:
        - observation: InventoryObservation
        - reward: RewardSchema (from last step)
        - done: bool
    """
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        state_data = env.state()
        return state_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)