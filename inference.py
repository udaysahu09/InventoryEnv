import os
import json
import requests
from openai import OpenAI
from models import TaskType, InventoryAction


def main():
    """Run inference on InventoryEnv with OpenAI LLM."""
    
    # Fetch configuration from environment
    api_base_url = os.environ.get("API_BASE_URL", "http://localhost:7860")
    model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    hf_token = os.environ.get("HF_TOKEN")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=hf_token if hf_token else "dummy-key",
        base_url=api_base_url if hf_token else None,
    )
    
    tasks = [TaskType.EASY, TaskType.MEDIUM, TaskType.HARD]
    
    for task in tasks:
        print(f"[START] Task: {task.value}")
        
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
            print(f"[ERROR] Failed to reset environment: {e}")
            continue
        
        reset_data = reset_response.json()
        observation = reset_data.get("observation", {})
        
        max_steps = 100
        final_score = 0.0
        
        for step in range(1, max_steps + 1):
            # Prepare state for LLM
            state_dict = observation
            
            # Create LLM prompt
            system_prompt = """You are an expert supply chain manager. Given the current warehouse state,
            decide how many units to order for each product to maximize profit and fulfill customer orders.
            Return ONLY a JSON object with key 'order_quantities' containing a list of integers."""
            
            user_prompt = f"""Current warehouse state:
            - Day: {state_dict.get('current_day', 0)}
            - Stock levels: {state_dict.get('stock_levels', [])}
            - Pending orders: {state_dict.get('pending_orders', [])}
            - Warehouse capacity: {state_dict.get('warehouse_capacity', 0)}
            - Warehouse used: {state_dict.get('warehouse_used', 0)}
            - Total fulfilled: {state_dict.get('total_fulfilled', 0)} / {state_dict.get('total_demand', 0)}
            
            Decide order quantities for each product (non-negative integers only)."""
            
            try:
                # Call LLM
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=100,
                )
                
                # Parse LLM response
                llm_output = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                try:
                    action_data = json.loads(llm_output)
                except json.JSONDecodeError:
                    # Fallback: extract JSON from text
                    import re
                    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                    if json_match:
                        action_data = json.loads(json_match.group())
                    else:
                        action_data = {"order_quantities": [0] * len(state_dict.get('stock_levels', []))}
                
                # Ensure action has correct format
                if "order_quantities" not in action_data:
                    action_data["order_quantities"] = [0] * len(state_dict.get('stock_levels', []))
                
                # Clamp to non-negative integers
                action_data["order_quantities"] = [
                    max(0, int(q)) for q in action_data["order_quantities"]
                ]
                
            except Exception as e:
                # Fallback action: no orders
                action_data = {"order_quantities": [0] * len(state_dict.get('stock_levels', []))}
            
            action_dict = action_data
            
            # Log step
            print(f"[STEP] State: {state_dict} | Action: {action_dict}")
            
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
                print(f"[ERROR] Failed to execute step: {e}")
                break
            
            step_data = step_response.json()
            observation = step_data.get("observation", {})
            reward = step_data.get("reward", {})
            done = step_data.get("done", False)
            
            final_score = reward.get("reward", 0.0)
            
            if done:
                break
        
        print(f"[END] Task: {task.value} | Final Score: {round(final_score, 4)}")
        print()


if __name__ == "__main__":
    main()