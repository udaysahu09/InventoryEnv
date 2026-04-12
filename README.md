---
title: InventoryEnv
emoji: 📦
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
app_file: app.py
pinned: false
---

# InventoryEnv - B2B Supply Chain RL Environment

Supply chain optimization using Reinforcement Learning for Meta Scaler Hackathon.

## Features

- 3 difficulty levels (easy, medium, hard)
- FastAPI server on port 7860
- Docker ready
- Interactive API docs at /docs

## Quick Start

### Local

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 7860
# InventoryEnv: B2B Supply Chain & Inventory Management

A production-ready OpenEnv environment for the Meta Scaler RL Hackathon. This environment simulates a B2B e-commerce warehouse where an RL agent manages inventory levels, fulfills customer orders, and optimizes warehouse capacity.

## 🎯 Overview

**InventoryEnv** challenges RL agents to manage a dynamic supply chain with multiple products, varying demand patterns, and operational constraints. The agent's goal is to maximize profit while maintaining high order fulfillment rates.

### Key Features
- ✅ **Three Difficulty Levels**: Easy (1 product), Medium (3 products, capacity limits), Hard (3 products, seasonal demand, supplier delays)
- ✅ **Realistic Constraints**: Warehouse capacity limits, storage costs, order fulfillment delays
- ✅ **Normalized Rewards**: Rewards strictly clamped to [0.0, 1.0] for consistent evaluation
- ✅ **FastAPI Integration**: RESTful endpoints for reset, step, and state queries
- ✅ **Pydantic Models**: Type-safe request/response schemas
- ✅ **Docker Ready**: Deployable to Hugging Face Spaces on port 7860

---

## 📊 Environment Specification

### Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `current_day` | int | Current simulation day (0 to max_days) |
| `stock_levels` | List[int] | Current inventory for each product |
| `warehouse_capacity` | int | Total warehouse capacity |
| `warehouse_used` | int | Current warehouse space used |
| `pending_orders` | List[int] | Unfulfilled orders per product |
| `total_fulfilled` | int | Total orders fulfilled so far |
| `total_demand` | int | Total order demand generated so far |
| `current_balance` | float | Account balance (revenue - costs) |
| `task` | str | Current task level (easy/medium/hard) |

**Example:**
```json
{
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
```

### Action Space

The agent outputs order quantities for each product:

| Field | Type | Constraints |
|-------|------|-------------|
| `order_quantities` | List[int] | Non-negative integers, one per product |

**Example:**
```json
{
  "order_quantities": [20, 15, 10]
}
```

### Reward Space

Structured reward with detailed breakdown:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `reward` | float | [0.0, 1.0] | Normalized reward (primary metric) |
| `fulfilled_percentage` | float | [0.0, 100.0] | Percentage of orders fulfilled |
| `oos_penalty` | float | [0.0, 0.2] | Out-of-stock penalty |
| `storage_cost` | float | [0.0, ∞) | Storage cost for current inventory |
| `order_cost` | float | [0.0, ∞) | Cost of orders placed |
| `total_revenue` | float | (-∞, ∞) | Total revenue (balance) |
| `done` | bool | True/False | Episode completion flag |

---

## 🎮 Tasks

### Easy Task
**Objective:** Manage 1 product with fixed demand. Prevent Out of Stock (OOS).

- **Duration:** 100 days
- **Products:** 1 (fixed demand ~5 units/day)
- **Warehouse Capacity:** 200 units
- **Complexity:** Low (no capacity constraints, fixed demand)
- **Success Metric:** Fulfill ≥ 90% of orders
- **Baseline Reward:** ~0.85-0.95

**Strategy:** Order consistently to match demand, prevent stockouts.

---

### Medium Task
**Objective:** Manage 3 products with strict warehouse capacity (300 units).

- **Duration:** 100 days
- **Products:** 3 (varying demand: 3-8 units/day)
- **Warehouse Capacity:** 300 units (shared)
- **Complexity:** Medium (capacity constraints, multi-product optimization)
- **Success Metric:** Fulfill ≥ 85% of orders, minimize storage costs
- **Baseline Reward:** ~0.75-0.85

**Product Details:**
- Product A: Base demand 5/day, Price $50, Storage cost $1/unit, Order cost $5
- Product B: Base demand 8/day, Price $75, Storage cost $1.5/unit, Order cost $7
- Product C: Base demand 3/day, Price $100, Storage cost $2/unit, Order cost $10

**Strategy:** Balance inventory across products, prioritize high-margin items, monitor warehouse capacity.

---

### Hard Task
**Objective:** Maximize profit with seasonal demand and supplier delays.

- **Duration:** 100 days
- **Products:** 3 (seasonal demand variation)
- **Warehouse Capacity:** 300 units (shared)
- **Demand Pattern:** Peak at days 30-40 and 70-80 (1.5x multiplier)
- **Supplier Delays:** 20% chance of 1-3 day delay per order
- **Complexity:** High (dynamic demand, supply chain delays, profit optimization)
- **Success Metric:** Maximize profit while maintaining high fulfillment
- **Baseline Reward:** ~0.65-0.80

**Strategy:** Anticipate seasonal peaks, maintain safety stock for delayed orders, optimize order timing and quantities.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip or conda
- Docker (for containerized deployment)

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/InventoryEnv.git
cd InventoryEnv
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the FastAPI server:**
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

The API will be available at `http://localhost:7860`

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t inventoryenv:latest .
```

2. **Run the container:**
```bash
docker run -d -p 7860:7860 --name inventoryenv inventoryenv:latest
```

3. **Verify health:**
```bash
curl http://localhost:7860/
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Upload all files to the Space repository
3. Set the Docker container option to port 7860
4. The environment will automatically build and deploy

---

## 📡 API Endpoints

### 1. Health Check
```bash
GET /
```
**Response:** 200 OK
```json
{
  "status": "ok",
  "message": "InventoryEnv API is running",
  "version": "1.0.0"
}
```

### 2. Reset Environment
```bash
POST /reset
Content-Type: application/json

{
  "task": "easy"
}
```

**Response:**
```json
{
  "observation": {
    "current_day": 0,
    "stock_levels": [50],
    "warehouse_capacity": 200,
    "warehouse_used": 50,
    "pending_orders": [0],
    "total_fulfilled": 0,
    "total_demand": 0,
    "current_balance": 0.0,
    "task": "easy"
  },
  "message": "Environment reset with task: easy"
}
```

### 3. Execute Step
```bash
POST /step
Content-Type: application/json

{
  "action": {
    "order_quantities": [20, 15, 10]
  }
}
```

**Response:**
```json
{
  "observation": { ... },
  "reward": {
    "reward": 0.85,
    "fulfilled_percentage": 85.0,
    "oos_penalty": 0.05,
    "storage_cost": 50.0,
    "order_cost": 100.0,
    "total_revenue": 500.0,
    "done": false
  },
  "done": false
}
```

### 4. Get State
```bash
GET /state
```

**Response:**
```json
{
  "observation": { ... },
  "reward": { ... },
  "done": false
}
```

---

## 🤖 Running the LLM Baseline

The `inference.py` script runs the environment using an OpenAI-compatible LLM.

### Setup
```bash
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your-huggingface-token"
```

### Run Baseline
```bash
python inference.py
```

**Expected Output:**
```
[START] Task: easy
[STEP] State: {...} | Action: {"order_quantities": [20]}
[STEP] State: {...} | Action: {"order_quantities": [18]}
...
[END] Task: easy | Final Score: 0.8745
```

---

## 📊 Reward Calculation

The reward is computed as follows:

1. **Fulfillment Reward:** `fulfilled_orders / total_demand` (normalized to [0, 1])
2. **Out-of-Stock Penalty:** Up to 0.2 based on number of products with pending orders
3. **Overstock Penalty:** 0.1 if warehouse > 80% capacity
4. **Hard Task:** Blends fulfillment (60%) + profit maximization (40%)

**Final Reward:** `max(0.0, min(1.0, fulfillment_reward - oos_penalty - overstock_penalty))`

---

## 📈 Baseline Performance

Expected baseline scores for a simple heuristic agent (ordering to match average demand):

| Task | Expected Reward | Notes |
|------|-----------------|-------|
| Easy | 0.88 | High fulfillment, minimal penalties |
| Medium | 0.72 | Balancing multi-product inventory |
| Hard | 0.65 | Challenging with seasonal spikes and delays |

---

## 🔧 Troubleshooting

### Server won't start
```bash
lsof -i :7860
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### LLM inference fails
- Verify API_BASE_URL is correct
- Check MODEL_NAME is available
- Ensure HF_TOKEN is set

### Out of memory
- Reduce max_steps in environment config
- Lower batch size if running multiple agents

---

## 📁 Project Structure

```
InventoryEnv/
├── models.py              # Pydantic data models
├── environment.py         # Core environment logic
├── app.py                 # FastAPI server
├── inference.py           # LLM baseline script
├── openenv.yaml           # OpenEnv specification
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

---

## 📝 License

MIT License

---

## 🎓 Citation

If you use InventoryEnv in your research:

```bibtex
@software{inventoryenv2024,
  title={InventoryEnv: B2B Supply Chain & Inventory Management},
  author={Meta Scaler Hackathon Participant},
  year={2024},
  url={https://github.com/yourusername/InventoryEnv}
}
```

---
