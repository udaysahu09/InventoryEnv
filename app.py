from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class TaskType(str, Enum):
    """Enum for different task difficulties"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class InventoryObservation(BaseModel):
    """
    Observation schema for the InventoryEnv.
    Represents the current state of the warehouse.
    """
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
    """
    Action schema for the InventoryEnv.
    Represents the quantity to order for each product.
    """
    order_quantities: List[int] = Field(
        ...,
        description="Quantity to order for each product. Negative values not allowed."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "order_quantities": [20, 15, 10]
            }
        }


class RewardSchema(BaseModel):
    """
    Reward schema for the InventoryEnv.
    Returns normalized reward and detailed breakdown.
    """
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
    """Request schema for reset endpoint - matches openenv.yaml specification"""
    task: str = Field(default="easy", description="Task difficulty level (easy, medium, hard)")

    def get_task(self) -> TaskType:
        """Extract and validate task from request."""
        try:
            return TaskType(self.task.lower())
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
