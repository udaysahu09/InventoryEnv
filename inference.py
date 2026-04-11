"""
InventoryEnv Inference Script
Runs all three tasks (Easy, Medium, Hard) with a smart inventory management strategy.
No LLM required - uses heuristic-based decision making.
"""

import os
import requests
from typing import Dict, List, Any
from models import TaskType


def smart_inventory_decision(state_dict: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Smart inventory management logic - heuristic-based ordering strategy.
    
    Makes intelligent ordering decisions based on warehouse state metrics.
    Prioritizes preventing stockouts while managing warehouse capacity.
    
    Args:
        state_dict (dict): Current environment state containing:
            - stock_levels: Current inventory for each product
            - pending_orders: Unfulfilled orders for each product
            - warehouse_capacity: Total warehouse capacity
            - warehouse_used: Current warehouse space used
            - total_fulfilled: Total orders fulfilled so far
            - total_demand: Total orders generated so far
            - current_day: Current simulation day
    
    Returns:
        dict: Action with order_quantities for each product
        
    Examples:
        >>> state = {
        ...     'stock_levels': [15],
        ...     'pending_orders': [5],
        ...     'warehouse_capacity': 200,
        ...     'warehouse_used': 50,
        ...     'total_fulfilled': 100,
        ...     'total_demand': 120
        ... }
        >>> smart_inventory_decision(state)
        {'order_quantities': [50]}
    """
    # Extract state variables with safe defaults
    stock_levels = state_dict.get('stock_levels', [50])
    pending_orders = state_dict.get('pending_orders', [0])
    warehouse_capacity = state_dict.get('warehouse_capacity', 200)
    warehouse_used = state_dict.get('warehouse_used', 50)
    total_fulfilled = state_dict.get('total_fulfilled', 0)
    total_demand = state_dict.get('total_demand', 0)
    
    # Get first product metrics (for easy task - single product)
    stock = stock_levels[0] if stock_levels else 0
    pending = pending_orders[0] if pending_orders else 0
    
    # Calculate key metrics
    available_capacity = warehouse_capacity - warehouse_used
    fulfillment_rate = total_fulfilled / max(total_demand, 1)
    
    # Initialize order quantity
    order_qty = 0
    
    # Decision Logic - Priority-based ordering strategy
    
    # Rule 1: CRITICAL - Stock critically low, order immediately
    if stock < 10:
        order_qty = min(100, available_capacity)
    
    # Rule 2: LOW STOCK - Stock low with pending demand
    elif stock < 20 and pending > 5:
        order_qty = min(80, available_capacity)
    
    # Rule 3: MODERATE STOCK - Moderate stock but high unfulfilled orders
    elif stock < 30 and pending > 10:
        order_qty = min(60, available_capacity)
    
    # Rule 4: PREVENTIVE ORDERING - Maintain safety stock
    elif stock < 30 and available_capacity > 40:
        order_qty = min(50, available_capacity)
    
    # Rule 5: LOW FULFILLMENT - Poor fulfillment rate, build inventory
    elif fulfillment_rate < 0.8 and stock < 40:
        order_qty = min(70, available_capacity)
    
    # Rule 6: GOOD STATUS - Stock sufficient, no ordering needed
    else:
        order_qty = 0
    
    return {"order_quantities": [int(order_qty)]}


def run_task(api_base_url: str, task: TaskType) -> Dict[str, Any]:
    """
    Execute a complete task episode with smart inventory management.
    
    Args:
        api_base_url (str): Base URL of the InventoryEnv API
        task (TaskType): Task difficulty level (easy, medium, hard)
    
    Returns:
        dict: Task results containing score, fulfillment, revenue, and days
        
    Raises:
        requests.RequestException: If API calls fail
    """
    print(f"\n{'='*70}")
    print(f"📦 TASK: {task.value.upper()}")
    print(f"{'='*70}\n")
    
    # Reset environment
    reset_payload = {"task": task.value}
    try:
        reset_response = requests.post(
            f"{api_base_url}/reset",
            json=reset_payload,
            timeout=10
        )
        reset_response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ [ERROR] Failed to reset environment: {e}")
        raise
    
    # Get initial observation
    reset_data = reset_response.json()
    observation = reset_data.get("observation", {})
    
    # Initialize tracking variables
    max_steps = 100
    final_score = 0.0
    total_fulfilled = 0
    total_demand = 0
    total_revenue = 0.0
    current_day = 0
    
    # Print table header
    print(f"{'Day':<6} {'Stock':<8} {'Pending':<10} {'Fulfilled':<12} {'Revenue':<12} {'Action':<15}")
    print("-" * 70)
    
    # Execute episode steps
    for step in range(1, max_steps + 1):
        state_dict = observation
        
        # Get smart decision
        action_dict = smart_inventory_decision(state_dict)
        
        # Extract current metrics for display
        current_day = state_dict.get('current_day', 0)
        stock = state_dict.get('stock_levels', [0])[0]
        pending = state_dict.get('pending_orders', [0])[0]
        fulfilled = state_dict.get('total_fulfilled', 0)
        revenue = state_dict.get('current_balance', 0)
        order = action_dict['order_quantities'][0]
        
        # Log current step
        print(f"{current_day:<6} {stock:<8} {pending:<10} {fulfilled:<12} ${revenue:<11.2f} Order: {order}")
        
        # Execute step in environment
        step_payload = {"action": action_dict}
        try:
            step_response = requests.post(
                f"{api_base_url}/step",
                json=step_payload,
                timeout=10
            )
            step_response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ [ERROR] Failed to execute step: {e}")
            break
        
        # Process response
        step_data = step_response.json()
        observation = step_data.get("observation", {})
        reward = step_data.get("reward", {})
        done = step_data.get("done", False)
        
        # Update tracking variables
        final_score = reward.get("reward", 0.0)
        total_fulfilled = observation.get('total_fulfilled', 0)
        total_demand = observation.get('total_demand', 0)
        total_revenue = observation.get('current_balance', 0)
        
        # Stop if episode is complete
        if done:
            break
    
    # Calculate final metrics
    fulfillment_rate = (total_fulfilled / max(total_demand, 1)) * 100
    
    # Print task results
    print("-" * 70)
    print(f"\n✅ TASK COMPLETE: {task.value.upper()}")
    print(f"  📊 Final Score: {round(final_score, 2)}")
    print(f"  📦 Total Fulfilled: {total_fulfilled}/{total_demand} ({fulfillment_rate:.1f}%)")
    print(f"  💰 Total Revenue: ${total_revenue:.2f}")
    print(f"  ⏱️  Days Completed: {current_day}")
    
    # Return results dictionary
    return {
        "score": round(final_score, 2),
        "fulfilled": f"{total_fulfilled}/{total_demand}",
        "fulfillment_rate": f"{fulfillment_rate:.1f}%",
        "revenue": f"${total_revenue:.2f}",
        "days": current_day
    }


def main():
    """
    Main entry point - runs all tasks (Easy, Medium, Hard) sequentially.
    """
    # Get API configuration
    api_base_url = os.environ.get("API_BASE_URL", "http://localhost:7860")
    
    # Print header
    print("=" * 70)
    print("🚀 InventoryEnv - Reinforcement Learning Environment")
    print("=" * 70)
    print(f"📍 API URL: {api_base_url}")
    print(f"🤖 Strategy: Smart Inventory Management (Heuristic-based)")
    print("=" * 70)
    
    # Run all tasks
    tasks = [TaskType.EASY, TaskType.MEDIUM, TaskType.HARD]
    results = {}
    
    for task in tasks:
        try:
            task_results = run_task(api_base_url, task)
            results[task.value] = task_results
        except Exception as e:
            print(f"❌ Failed to complete task {task.value}: {e}")
            results[task.value] = {
                "score": 0.0,
                "fulfilled": "0/0",
                "fulfillment_rate": "0.0%",
                "revenue": "$0.00",
                "days": 0
            }
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("📋 SUMMARY - ALL TASKS")
    print(f"{'='*70}\n")
    
    for task_name, metrics in results.items():
        print(f"🎯 {task_name.upper()}")
        for key, value in metrics.items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")
        print()
    
    print("=" * 70)
    print("✨ InventoryEnv Inference Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
