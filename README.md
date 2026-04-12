---
title: InventoryEnv
emoji: 📦
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
app_file: server/app.py
pinned: false
---

# InventoryEnv - B2B Supply Chain RL Environment

**A production-ready OpenEnv environment for the Meta Scaler RL Hackathon**

Supply chain optimization using Reinforcement Learning for intelligent inventory management in B2B e-commerce.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-brightgreen)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

**InventoryEnv** is a realistic B2B supply chain simulation where RL agents manage warehouse inventory, fulfill customer orders, and optimize profit. The environment provides three difficulty levels with increasing complexity: from simple single-product management to complex multi-product scenarios with seasonal demand and supplier delays.

### Key Features

- ✅ **Three Difficulty Levels**
  - **Easy**: Manage 1 product with fixed demand, prevent out-of-stock (OOS)
  - **Medium**: Manage 3 products with warehouse capacity constraints
  - **Hard**: Manage 3 products with seasonal demand spikes and supplier delays

- ✅ **Realistic Constraints**
  - Warehouse capacity limits
  - Storage costs for inventory
  - Order fulfillment costs
  - Dynamic demand patterns

- ✅ **Normalized Rewards**
  - Strictly bounded to [0.0, 1.0] for consistent evaluation
  - Detailed reward breakdown (fulfillment %, penalties, costs, revenue)
  - Easy integration with RL frameworks

- ✅ **FastAPI Integration**
  - RESTful API endpoints (reset, step, state, health check)
  - Async/await support for high performance
  - Interactive API docs at `/docs`
  - Request/response validation with Pydantic

- ✅ **Production Ready**
  - Docker containerized
  - Deployable to Hugging Face Spaces
  - Full error handling and logging
  - Type hints throughout

---

## 📊 Environment Specification

### Observation Space

Each observation represents the current warehouse state:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `current_day` | int | [0, max_days] | Current simulation day |
| `stock_levels` | List[int] | [0, ∞) | Current inventory per product |
| `warehouse_capacity` | int | Fixed | Total warehouse capacity |
| `warehouse_used` | int | [0, capacity] | Current space used |
| `pending_orders` | List[int] | [0, ∞) | Unfulfilled orders per product |
| `total_fulfilled` | int | [0, ∞) | Total orders fulfilled lifetime |
| `total_demand` | int | [0, ∞) | Total demand generated lifetime |
| `current_balance` | float | (-∞, ∞) | Account balance (revenue - costs) |
| `task` | str | {easy, medium, hard} | Current task level |

**Example Observation:**
```json
{
  "current_day": 5,
  "stock_levels": [50, 30, 20],
  "warehouse_capacity": 300,
  "warehouse_used": 100,
  "pending_orders": [10, 5, 8],
  "total_fulfilled": 45,
  "total_demand": 60,
  "current_balance": 1250.50,
  "task": "medium"
}
