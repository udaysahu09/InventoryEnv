import os
import json
import requests
from dotenv import load_dotenv
from models import TaskType, InventoryAction

# Load environment variables from .env file
load_dotenv()


def smart_inventory_decision(state_dict):
    """
    Smart inventory management logic without LLM.
    Makes intelligent ordering decisions based on warehouse state.
    """
    stock_levels = state_dict.get('stock_levels', [50])
    pending_orders = state_dict.get('pending_orders', [0])
    warehouse_capacity = state_dict.get('warehouse_capacity', 200)
    warehouse_used = state_dict.get('warehouse_used', 50)
    total_fulfilled = state_dict.get('total_fulfilled', 0)
    total_demand = state_dict.get('total_demand', 0)
    current_day = state_dict.get('current_day', 0)
    
    stock = stock_levels[0] if stock_levels else 0
    pending = pending_orders[0] if pending_orders else 0
    
    # Calculate metrics
    available_capacity = warehouse_capacity - warehouse_used
    fulfillment_rate = total_fulfilled / max(total_demand, 1)
    
    # Decision logic
    order_qty = 0
    
    # Rule 1: Critical stock - order immediately
    if stock < 10:
        order_qty = min(100, available_capacity)
    
    # Rule 2: Low stock with pending orders
    elif stock < 20 and pending > 5:
        order_qty = min(80, available_capacity)
    
    # Rule 3: Moderate stock but high pending
    elif stock < 30 and pending > 10:
        order_qty = min(60, available_capacity)
    
    # Rule 4: Preventive ordering (maintain 30-50 stock)
    elif stock < 30 and available_capacity > 40:
        order_qty = min(50, available_capacity)
    
    # Rule 5: Low fulfillment rate - stock up
    elif fulfillment_rate < 0.8 and stock < 40:
        order_qty = min(70, available_capacity)
    
    # Rule 6: Good stock, maintain status
    else:
        order_qty = 0
    
    return {"order_quantities": [int(order_qty)]}


def main():
    """Run InventoryEnv with smart inventory management."""
    
    # Fetch configuration from environment
    api_base_url = os.environ.get("API_BASE_URL", "http://localhost:7860")
    
    print("=" * 70)
    print("🚀 InventoryEnv - Reinforcement Learning Environment")
    print("=" * 70)
    print(f"📍 API URL: {api_base_url}")
    print(f"🤖 Strategy: Smart Inventory Management (No LLM Required)")
    print("=" * 70)
    print()
    
    tasks = [TaskType.EASY, TaskType.MEDIUM, TaskType.HARD]
    results = {}
    
    for task in tasks:
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
            continue
        
        reset_data = reset_response.json()
        observation = reset_data.get("observation", {})
        
        max_steps = 100
        final_score = 0.0
        total_fulfilled = 0
        total_demand = 0
        total_revenue = 0
        
        print(f"{'Day':<6} {'Stock':<8} {'Pending':<10} {'Fulfilled':<12} {'Revenue':<12} {'Action':<15}")
        print("-" * 70)
        
        for step in range(1, max_steps + 1):
            state_dict = observation
            
            # Get smart decision
            action_dict = smart_inventory_decision(state_dict)
            
            current_day = state_dict.get('current_day', 0)
            stock = state_dict.get('stock_levels', [0])[0]
            pending = state_dict.get('pending_orders', [0])[0]
            fulfilled = state_dict.get('total_fulfilled', 0)
            revenue = state_dict.get('current_balance', 0)
            order = action_dict['order_quantities'][0]
            
            print(f"{current_day:<6} {stock:<8} {pending:<10} {fulfilled:<12} ${revenue:<11.2f} Order: {order}")
            
            # Execute step
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
            
            step_data = step_response.json()
            observation = step_data.get("observation", {})
            reward = step_data.get("reward", {})
            done = step_data.get("done", False)
            
            final_score = reward.get("reward", 0.0)
            total_fulfilled = observation.get('total_fulfilled', 0)
            total_demand = observation.get('total_demand', 0)
            total_revenue = observation.get('current_balance', 0)
            
            if done:
                break
        
        # Calculate metrics
        fulfillment_rate = (total_fulfilled / max(total_demand, 1)) * 100
        
        print("-" * 70)
        print(f"\n✅ TASK COMPLETE: {task.value.upper()}")
        print(f"  📊 Final Score: {round(final_score, 2)}")
        print(f"  📦 Total Fulfilled: {total_fulfilled}/{total_demand} ({fulfillment_rate:.1f}%)")
        print(f"  💰 Total Revenue: ${total_revenue:.2f}")
        print(f"  ⏱️  Days Completed: {current_day}")
        
        results[task.value] = {
            "score": round(final_score, 2),
            "fulfilled": f"{total_fulfilled}/{total_demand}",
            "fulfillment_rate": f"{fulfillment_rate:.1f}%",
            "revenue": f"${total_revenue:.2f}",
            "days": current_day
        }
    
    # Summary
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