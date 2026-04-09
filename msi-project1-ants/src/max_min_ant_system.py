import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.config import AlgorithmConfig
from src.cvrp_instance import CVRPInstance


@dataclass
class MMASResult:
    routes: List[List[int]]
    total_length: float
    feasible: bool
    best_iteration: int


class MaxMinAntSystemSolver:
    def __init__(self, instance: CVRPInstance, config: AlgorithmConfig):
        self.instance = instance
        self.config = config
        self.rng = random.Random(config.seed)

        self.nodes = sorted(instance.node_coords.keys())
        self.customers = sorted(instance.customers())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        self.distance_matrix = self._build_distance_matrix()
        self.heuristic_matrix = self._build_heuristic_matrix()

        self.tau_max, self.tau_min = self._compute_initial_tau_bounds()
        self.pheromone_matrix = self._initialize_pheromones()

    def _compute_initial_tau_bounds(self):
        """
        Początkowe ograniczenia feromonu wyznaczane na podstawie rozwiązania zachłannego.
        """
        from src.greedy import solve_greedy

        greedy_result = solve_greedy(self.instance)

        rho = self.config.evaporation
        n = max(1, len(self.customers))

        if greedy_result.feasible and greedy_result.total_length > 0 and rho > 0:
            tau_max = 1.0 / (rho * greedy_result.total_length)
            tau_min = tau_max / (2.0 * n)
            return tau_max, tau_min

        return 1.0, 1.0 / (2.0 * n)
    
    def _build_distance_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        distances = np.zeros((n, n), dtype=float)

        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i == j:
                    distances[i, j] = 0.0
                else:
                    distances[i, j] = self.instance.distance(node_i, node_j)

        return distances

    def _build_heuristic_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        heuristic = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                d = self.distance_matrix[i, j]
                if i != j and d > 0:
                    heuristic[i, j] = 1.0 / d

        return heuristic

    def _initialize_pheromones(self) -> np.ndarray:
        n = len(self.nodes)
        return np.full((n, n), self.tau_max, dtype=float)

    def _choose_next_customer(
        self,
        current_node: int,
        remaining_customers: List[int],
        current_route: List[int],
    ) -> Optional[int]:
        feasible_customers = []

        for customer in remaining_customers:
            candidate_route = current_route + [customer]
            if self.instance.is_route_feasible(candidate_route):
                feasible_customers.append(customer)

        if not feasible_customers:
            return None

        current_idx = self.node_to_idx[current_node]
        weights = []

        for customer in feasible_customers:
            customer_idx = self.node_to_idx[customer]
            tau = self.pheromone_matrix[current_idx, customer_idx]
            eta = self.heuristic_matrix[current_idx, customer_idx]
            value = (tau ** self.config.alpha) * (eta ** self.config.beta)
            weights.append(value)

        total_weight = sum(weights)

        if total_weight == 0:
            return self.rng.choice(feasible_customers)

        probabilities = [w / total_weight for w in weights]
        return self.rng.choices(feasible_customers, weights=probabilities, k=1)[0]

    def _construct_solution(self) -> List[List[int]]:
        remaining_customers = self.customers.copy()
        routes: List[List[int]] = []

        while remaining_customers:
            if len(routes) >= self.instance.num_vehicles:
                return []

            current_route: List[int] = []
            current_node = self.instance.depot_id

            while True:
                next_customer = self._choose_next_customer(
                    current_node=current_node,
                    remaining_customers=remaining_customers,
                    current_route=current_route,
                )

                if next_customer is None:
                    break

                current_route.append(next_customer)
                remaining_customers.remove(next_customer)
                current_node = next_customer

            if not current_route:
                return []

            routes.append(current_route)

        return routes

    def _evaporate_pheromones(self) -> None:
        self.pheromone_matrix *= (1.0 - self.config.evaporation)

    def _deposit_best_solution(self, routes: List[List[int]], total_length: float) -> None:
        if total_length <= 0:
            return

        delta = self.config.q / total_length

        for route in routes:
            full_route = [self.instance.depot_id] + route + [self.instance.depot_id]

            for i in range(len(full_route) - 1):
                a = full_route[i]
                b = full_route[i + 1]
                a_idx = self.node_to_idx[a]
                b_idx = self.node_to_idx[b]

                self.pheromone_matrix[a_idx, b_idx] += delta
                self.pheromone_matrix[b_idx, a_idx] += delta

    def _clip_pheromones(self) -> None:
        self.pheromone_matrix = np.clip(self.pheromone_matrix, self.tau_min, self.tau_max)

    def _update_tau_bounds(self, best_length: float) -> None:
        """
        Aktualizacja tau_max i tau_min zgodnie z ideą MMAS.
        """
        if best_length <= 0 or self.config.evaporation <= 0:
            return

        rho = self.config.evaporation
        n = max(1, len(self.customers))

        self.tau_max = 1.0 / (rho * best_length)
        self.tau_min = self.tau_max / (2.0 * n)
        
    def solve(self) -> MMASResult:
        best_routes: List[List[int]] = []
        best_length = float("inf")
        best_iteration = -1

        for iteration in range(self.config.iterations):
            iteration_best_routes: List[List[int]] = []
            iteration_best_length = float("inf")

            for _ in range(self.config.ants):
                routes = self._construct_solution()

                if not routes:
                    continue

                if not self.instance.is_solution_feasible(routes):
                    continue

                total_length = self.instance.solution_length(routes)

                if total_length < iteration_best_length:
                    iteration_best_length = total_length
                    iteration_best_routes = routes

                if total_length < best_length:
                    best_length = total_length
                    best_routes = routes
                    best_iteration = iteration + 1

                if best_routes:
                    self._update_tau_bounds(best_length)

            self._evaporate_pheromones()

            if iteration_best_routes:
                self._deposit_best_solution(iteration_best_routes, iteration_best_length)

            self._clip_pheromones()

        feasible = len(best_routes) > 0 and self.instance.is_solution_feasible(best_routes)

        return MMASResult(
            routes=best_routes,
            total_length=best_length if feasible else float("inf"),
            feasible=feasible,
            best_iteration=best_iteration,
        )