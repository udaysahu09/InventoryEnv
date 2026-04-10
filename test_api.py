import requests
import json

API_URL = "http://localhost:7860"

# Test 1: Health Check
print("=" * 50)
print("TEST 1: Health Check")
print("=" * 50)
response = requests.get(f"{API_URL}/")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test 2: Reset Easy Task
print("=" * 50)
print("TEST 2: Reset Easy Task")
print("=" * 50)
response = requests.post(f"{API_URL}/reset", json={"task": "easy"})
print(f"Status: {response.status_code}")
reset_data = response.json()
print(f"Response: {json.dumps(reset_data, indent=2)}\n")

# Test 3: Execute Step
print("=" * 50)
print("TEST 3: Execute Step (Order 10 units)")
print("=" * 50)
response = requests.post(f"{API_URL}/step", json={"action": {"order_quantities": [10]}})
print(f"Status: {response.status_code}")
step_data = response.json()
print(f"Observation: {step_data.get('observation')}")
print(f"Reward: {step_data.get('reward')}")
print(f"Done: {step_data.get('done')}\n")

# Test 4: Get State
print("=" * 50)
print("TEST 4: Get Current State")
print("=" * 50)
response = requests.get(f"{API_URL}/state")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

print("✅ All tests completed!")