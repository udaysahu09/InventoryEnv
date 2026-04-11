import random
from typing import Tuple, Dict, List
from app import InventoryObservation, InventoryAction, RewardSchema, TaskType


class InventoryEnv:
    """
    InventoryEnv: A B2B supply chain agent managing an e-commerce warehouse.
    Tasks:
    - Easy: Manage 1 product with fixed demand, prevent OOS
    - Medium: Manage 3 products with warehouse capacity limit, avoid overstock
    - Hard: Manage 3 products with seasonal demand and supplier delays, maximize profit
    """

    # Task configurations
    TASK_CONFIG = {
        TaskType.EASY: {
            "num_products": 1,
            "max_days": 100,
            "warehouse_capacity": 200,
            "products": [
                {
                    "name": "Product_A",
                    "base_demand": 5,
                    "price": 50.0,
                    "storage_cost": 1.0,
                    "order_cost": 5.0,
                }
            ],
            "seasonal": False,
            "supplier_delay": False,
        },
        TaskType.MEDIUM: {
            "num_products": 3,
            "max_days": 100,
            "warehouse_capacity": 300,
            "products": [
                {
                    "name": "Product_A",
                    "base_demand": 5,
                    "price": 50.0,
                    "storage_cost": 1.0,
                    "order_cost": 5.0,
                },
                {
                    "name": "Product_B",
                    "base_demand": 8,
                    "price": 75.0,
                    "storage_cost": 1.5,
                    "order_cost": 7.0,
                },
                {
                    "name": "Product_C",
                    "base_demand": 3,
                    "price": 100.0,
                    "storage_cost": 2.0,
                    "order_cost": 10.0,
                },
            ],
            "seasonal": False,
            "supplier_delay": False,
        },
        TaskType.HARD: {
            "num_products": 3,
            "max_days": 100,
            "warehouse_capacity": 300,
            "products": [
                {
                    "name": "Product_A",
                    "base_demand": 5,
                    "price": 50.0,
                    "storage_cost": 1.0,
                    "order_cost": 5.0,
                },
                {
                    "name": "Product_B",
                    "base_demand": 8,
                    "price": 75.0,
                    "storage_cost": 1.5,
                    "order_cost": 7.0,
                },
                {
                    "name": "Product_C",
                    "base_demand": 3,
                    "price": 100.0,
                    "storage_cost": 2.0,
                    "order_cost": 10.0,
                },
            ],
            "seasonal": True,
            "supplier_delay": True,
        },
    }

    def __init__(self, task: TaskType = TaskType.EASY):
        """Initialize the environment with a specific task."""
        self.task = task
        self.config = self.TASK_CONFIG[task]
        self.num_products = self.config["num_products"]
        self.max_days = self.config["max_days"]
        self.warehouse_capacity = self.config["warehouse_capacity"]
        self.products = self.config["products"]
        self.has_seasonal = self.config["seasonal"]
        self.has_supplier_delay = self.config["supplier_delay"]

        # State variables
        self.current_day = 0
        self.stock_levels = [50] * self.num_products  # Start with 50 units per product
        self.pending_orders = [0] * self.num_products
        self.total_fulfilled = 0
        self.total_demand = 0
        self.current_balance = 0.0
        self.last_reward = None
        self.last_done = False

        # For supplier delays (Hard task)
        self.pending_supplies = [0] * self.num_products
        self.supply_delay_counter = [0] * self.num_products

    def reset(self) -> InventoryObservation:
        """Reset the environment to initial state."""
        self.current_day = 0
        self.stock_levels = [50] * self.num_products
        self.pending_orders = [0] * self.num_products
        self.total_fulfilled = 0
        self.total_demand = 0
        self.current_balance = 0.0
        self.pending_supplies = [0] * self.num_products
        self.supply_delay_counter = [0] * self.num_products
        self.last_reward = None
        self.last_done = False

        return self._get_observation()

    def _get_observation(self) -> InventoryObservation:
        """Get the current observation."""
        warehouse_used = sum(self.stock_levels)
        return InventoryObservation(
            current_day=self.current_day,
            stock_levels=self.stock_levels.copy(),
            warehouse_capacity=self.warehouse_capacity,
            warehouse_used=warehouse_used,
            pending_orders=self.pending_orders.copy(),
            total_fulfilled=self.total_fulfilled,
            total_demand=self.total_demand,
            current_balance=round(self.current_balance, 2),
            task=self.task.value,
        )

    def _generate_demand(self) -> List[int]:
        """Generate demand for each product."""
        demand = []
        for product in self.products:
            base_demand = product["base_demand"]

            # Add seasonal variation (Hard task)
            if self.has_seasonal:
                # Seasonal pattern: peak at days 30-40 and 70-80
                if (30 <= self.current_day <= 40) or (70 <= self.current_day <= 80):
                    seasonal_factor = 1.5
                else:
                    seasonal_factor = 1.0
                base_demand = int(base_demand * seasonal_factor)

            # Add randomness
            demand.append(max(0, base_demand + random.randint(-2, 2)))

        return demand

    def _process_supplies(self):
        """Process pending supplies (Hard task with supplier delays)."""
        if not self.has_supplier_delay:
            return

        for i in range(self.num_products):
            if self.supply_delay_counter[i] > 0:
                self.supply_delay_counter[i] -= 1
                if self.supply_delay_counter[i] == 0:
                    # Supply arrived
                    self.stock_levels[i] += self.pending_supplies[i]
                    self.pending_supplies[i] = 0

    def _fulfill_orders(self, demand: List[int]) -> List[int]:
        """Fulfill pending orders and new demand."""
        fulfilled = [0] * self.num_products

        # First, add new demand to pending orders
        self.total_demand += sum(demand)
        for i in range(self.num_products):
            self.pending_orders[i] += demand[i]

        # Fulfill orders based on stock
        for i in range(self.num_products):
            if self.stock_levels[i] > 0 and self.pending_orders[i] > 0:
                can_fulfill = min(self.stock_levels[i], self.pending_orders[i])
                fulfilled[i] = can_fulfill
                self.stock_levels[i] -= can_fulfill
                self.pending_orders[i] -= can_fulfill
                self.total_fulfilled += can_fulfill
                # Revenue from fulfilled orders
                revenue = can_fulfill * self.products[i]["price"]
                self.current_balance += revenue

        return fulfilled

    def step(self, action: InventoryAction) -> Tuple[InventoryObservation, RewardSchema, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action: InventoryAction with order quantities
            
        Returns:
            Tuple of (observation, reward, done)
        """
        self.current_day += 1

        # Validate action
        order_quantities = action.order_quantities
        if len(order_quantities) != self.num_products:
            order_quantities = [0] * self.num_products

        # Clamp negative orders to 0
        order_quantities = [max(0, q) for q in order_quantities]

        # Process supplies (Hard task)
        self._process_supplies()

        # Apply orders
        total_order_cost = 0.0
        for i in range(self.num_products):
            quantity = order_quantities[i]
            if quantity > 0:
                order_cost = quantity * self.products[i]["order_cost"]
                total_order_cost += order_cost
                self.current_balance -= order_cost

                # Handle supplier delay (Hard task)
                if self.has_supplier_delay and random.random() < 0.2:  # 20% chance of delay
                    delay_days = random.randint(1, 3)
                    self.pending_supplies[i] = quantity
                    self.supply_delay_counter[i] = delay_days
                else:
                    self.stock_levels[i] += quantity

        # Generate demand and fulfill orders
        demand = self._generate_demand()
        fulfilled = self._fulfill_orders(demand)

        # Calculate storage costs
        storage_cost = 0.0
        for i in range(self.num_products):
            storage_cost += self.stock_levels[i] * self.products[i]["storage_cost"]
        self.current_balance -= storage_cost

        # Calculate reward
        reward_obj = self._calculate_reward(
            fulfilled, demand, storage_cost, total_order_cost
        )
        self.last_reward = reward_obj
        self.last_done = reward_obj.done

        observation = self._get_observation()

        return observation, reward_obj, reward_obj.done

    def _calculate_reward(
        self,
        fulfilled: List[int],
        demand: List[int],
        storage_cost: float,
        order_cost: float,
    ) -> RewardSchema:
        """Calculate normalized reward (strictly between 0.0 and 1.0)."""
        # Prevent division by zero
        if self.total_demand == 0:
            fulfilled_percentage = 100.0
        else:
            fulfilled_percentage = (self.total_fulfilled / self.total_demand) * 100.0
            fulfilled_percentage = min(100.0, fulfilled_percentage)

        # Out-of-stock penalty
        oos_count = sum(1 for p in self.pending_orders if p > 0)
        oos_penalty = (oos_count / self.num_products) * 0.2  # Max 0.2 penalty

        # Overstock penalty (Medium & Hard tasks)
        warehouse_used = sum(self.stock_levels)
        if warehouse_used > self.warehouse_capacity * 0.8:
            overstock_penalty = 0.1
        else:
            overstock_penalty = 0.0

        # Base reward from fulfillment
        fulfillment_reward = fulfilled_percentage / 100.0

        # Penalize for unfulfilled and storage costs (balance sheet)
        # For Hard task, also consider profit
        if self.task == TaskType.HARD:
            # Profit-based reward
            profit = self.current_balance
            profit_reward = min(1.0, max(0.0, profit / 500.0))  # Normalize to 500 as baseline
            reward = (fulfillment_reward * 0.6 + profit_reward * 0.4)
        else:
            reward = fulfillment_reward

        # Apply penalties
        reward = reward - oos_penalty - overstock_penalty
        
        # Clamp to strictly between 0 and 1 (exclusive bounds)
        if reward <= 0.0:
            reward = 0.01
        elif reward >= 1.0:
            reward = 0.99
        else:
            # Additional safeguard for floating-point precision
            if reward < 0.001:
                reward = 0.01
            elif reward > 0.999:
                reward = 0.99

        done = self.current_day >= self.max_days

        return RewardSchema(
            reward=round(reward, 4),
            fulfilled_percentage=round(fulfilled_percentage, 2),
            oos_penalty=round(oos_penalty, 4),
            storage_cost=round(storage_cost, 2),
            order_cost=round(order_cost, 2),
            total_revenue=round(self.current_balance, 2),
            done=done,
        )

    def state(self) -> Dict:
        """Get the current state as a dictionary."""
        obs = self._get_observation()
        return {
            "observation": obs.model_dump(),
            "reward": self.last_reward.model_dump() if self.last_reward else None,
            "done": self.last_done,
        }
