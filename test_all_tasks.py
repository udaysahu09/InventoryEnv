import requests
import json

API_URL = "http://localhost:7860"

def test_task(task_name):
    print("\n" + "=" * 60)
    print(f"TESTING TASK: {task_name.upper()}")
    print("=" * 60)
    
    # Reset
    response = requests.post(f"{API_URL}/reset", json={"task": task_name})
    obs = response.json()["observation"]
    print(f"\n📊 Initial State:")
    print(f"   Products: {len(obs['stock_levels'])}")
    print(f"   Stock: {obs['stock_levels']}")
    print(f"   Capacity: {obs['warehouse_capacity']}")
    
    # Run 5 steps
    total_reward = 0
    for step_num in range(1, 6):
        # Simple strategy: order based on pending orders
        order_qty = [max(5, p) for p in obs['pending_orders']]
        
        response = requests.post(
            f"{API_URL}/step", 
            json={"action": {"order_quantities": order_qty}}
        )
        
        data = response.json()
        obs = data["observation"]
        reward = data["reward"]
        total_reward += reward["reward"]
        
        print(f"\n   Step {step_num}:")
        print(f"      Day: {obs['current_day']}")
        print(f"      Stock: {obs['stock_levels']}")
        print(f"      Reward: {reward['reward']:.4f}")
        print(f"      Fulfilled: {reward['fulfilled_percentage']:.1f}%")
        print(f"      Balance: ${reward['total_revenue']:.2f}")
    
    avg_reward = total_reward / 5
    print(f"\n   📈 Average Reward (5 steps): {avg_reward:.4f}")
    return avg_reward

# Test all tasks
tasks = ["easy", "medium", "hard"]
results = {}

for task in tasks:
    try:
        avg = test_task(task)
        results[task] = avg
    except Exception as e:
        print(f"❌ Error in {task}: {e}")
        results[task] = 0.0

# Summary
print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)
for task, reward in results.items():
    print(f"{task.upper():10} | Avg Reward: {reward:.4f}")