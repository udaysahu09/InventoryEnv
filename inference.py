"""
Inference wrapper for InventoryEnv - Meta Scaler Compatible
Provides the validation interface Meta Scaler expects
"""

import json
import sys
from typing import Dict, List, Any, Tuple
from fastapi.testclient import TestClient
from app import app

# Create test client
client = TestClient(app)


class InventoryEnvInference:
    """Inference wrapper for InventoryEnv"""
    
    def __init__(self):
        """Initialize the inference engine"""
        self.env = None
        self.task = "easy"
    
    def reset(self, task: str = "easy") -> Dict[str, Any]:
        """
        Reset the environment
        
        Args:
            task: Task difficulty level (easy, medium, hard)
            
        Returns:
            dict: Initial observation
        """
        self.task = task
        response = client.post(
            "/reset",
            json={"task": task}
        )
        
        if response.status_code != 200:
            raise Exception(f"Reset failed: {response.text}")
        
        data = response.json()
        return {
            "observation": data.get("observation", {}),
            "message": data.get("message", "")
        }
    
    def step(self, action: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Execute one step
        
        Args:
            action: Action dictionary with order_quantities
            
        Returns:
            dict: observation, reward, done
        """
        response = client.post(
            "/step",
            json={"action": action}
        )
        
        if response.status_code != 200:
            raise Exception(f"Step failed: {response.text}")
        
        data = response.json()
        return {
            "observation": data.get("observation", {}),
            "reward": data.get("reward", {}),
            "done": data.get("done", False)
        }
    
    def state(self) -> Dict[str, Any]:
        """
        Get current state
        
        Returns:
            dict: Current state with observation, reward, done
        """
        response = client.get("/state")
        
        if response.status_code != 200:
            raise Exception(f"State failed: {response.text}")
        
        return response.json()


def validate_reset():
    """Validate /reset endpoint"""
    print("🧪 Testing POST /reset...")
    
    inference = InventoryEnvInference()
    
    # Test reset with easy task
    result = inference.reset(task="easy")
    assert "observation" in result, "Missing observation in reset response"
    obs = result["observation"]
    
    # Validate observation structure
    assert "current_day" in obs, "Missing current_day"
    assert "stock_levels" in obs, "Missing stock_levels"
    assert "warehouse_capacity" in obs, "Missing warehouse_capacity"
    assert "pending_orders" in obs, "Missing pending_orders"
    assert "total_fulfilled" in obs, "Missing total_fulfilled"
    assert "total_demand" in obs, "Missing total_demand"
    assert "current_balance" in obs, "Missing current_balance"
    assert "task" in obs, "Missing task"
    
    print("✅ POST /reset validation PASSED\n")
    return True


def validate_step():
    """Validate /step endpoint"""
    print("🧪 Testing POST /step...")
    
    inference = InventoryEnvInference()
    
    # Reset first
    inference.reset(task="easy")
    
    # Execute step
    action = {"order_quantities": [10]}
    result = inference.step(action)
    
    assert "observation" in result, "Missing observation in step response"
    assert "reward" in result, "Missing reward in step response"
    assert "done" in result, "Missing done in step response"
    
    # Validate reward structure
    reward = result["reward"]
    assert "reward" in reward, "Missing reward value"
    assert "fulfilled_percentage" in reward, "Missing fulfilled_percentage"
    assert "oos_penalty" in reward, "Missing oos_penalty"
    assert "storage_cost" in reward, "Missing storage_cost"
    assert "order_cost" in reward, "Missing order_cost"
    assert "total_revenue" in reward, "Missing total_revenue"
    assert "done" in reward, "Missing done in reward"
    
    # Validate reward bounds (strictly between 0 and 1)
    reward_val = reward["reward"]
    assert 0.0 < reward_val < 1.0, f"Reward {reward_val} not in (0.0, 1.0)"
    
    print("✅ POST /step validation PASSED\n")
    return True


def validate_state():
    """Validate /state endpoint"""
    print("🧪 Testing GET /state...")
    
    inference = InventoryEnvInference()
    
    # Reset first
    inference.reset(task="easy")
    
    # Get state
    result = inference.state()
    
    assert "observation" in result, "Missing observation in state response"
    assert "reward" in result, "Missing reward in state response"
    assert "done" in result, "Missing done in state response"
    
    print("✅ GET /state validation PASSED\n")
    return True


def validate_health_check():
    """Validate / health check endpoint"""
    print("🧪 Testing GET /...")
    
    response = client.get("/")
    assert response.status_code == 200, "Health check failed"
    
    data = response.json()
    assert "status" in data, "Missing status"
    assert data["status"] == "ok", "Status not ok"
    assert "message" in data, "Missing message"
    assert "version" in data, "Missing version"
    
    print("✅ GET / validation PASSED\n")
    return True


def main():
    """Run all validations"""
    print("\n" + "="*70)
    print("🔍 VALIDATING INVENTORYENV API - META SCALER COMPATIBILITY")
    print("="*70 + "\n")
    
    validations = [
        ("Health Check", validate_health_check),
        ("Reset Endpoint", validate_reset),
        ("Step Endpoint", validate_step),
        ("State Endpoint", validate_state),
    ]
    
    results = {}
    for name, validator in validations:
        try:
            validator()
            results[name] = "✅ PASSED"
        except Exception as e:
            print(f"❌ {name} FAILED: {e}\n")
            results[name] = f"❌ FAILED: {e}"
    
    # Print summary
    print("="*70)
    print("📋 VALIDATION SUMMARY")
    print("="*70 + "\n")
    
    for name, status in results.items():
        print(f"{name}: {status}")
    
    print("\n" + "="*70)
    
    # Check if all passed
    all_passed = all("✅" in status for status in results.values())
    if all_passed:
        print("✨ ALL VALIDATIONS PASSED - READY FOR META SCALER!")
    else:
        print("⚠️  SOME VALIDATIONS FAILED - PLEASE FIX")
    
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
