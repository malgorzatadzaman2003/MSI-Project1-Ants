import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.config import AlgorithmConfig
from src.cvrp_instance import CVRPInstance


@dataclass
class AntSystemResult:
    routes: List[List[int]]
    total_length: float
    feasible: bool
    best_iteration: int


class AntSystemSolver:
    def __init__(self, instance: CVRPInstance, config: AlgorithmConfig):
        self.instance = instance
        self.config = config
        self.rng = random.Random(config.seed)

        self.nodes = sorted(instance.node_coords.keys())
        self.customers = sorted(instance.customers())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}

        self.distance_matrix = self._build_distance_matrix()
        self.heuristic_matrix = self._build_heuristic_matrix()
        self.pheromone_matrix = self._initialize_pheromones()

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
        """
        Heurystyka eta_ij = 1 / distance(i, j).
        Dla i == j ustawiamy 0.
        """
        n = len(self.nodes)
        heuristic = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                d = self.distance_matrix[i, j]
                if i != j and d > 0:
                    heuristic[i, j] = 1.0 / d

        return heuristic

    def _initialize_pheromones(self) -> np.ndarray:
        """
        Początkowy feromon ustawiamy jako stałą dodatnią.
        """
        n = len(self.nodes)
        return np.full((n, n), 1.0, dtype=float)

    def _choose_next_customer(
        self,
        current_node: int,
        remaining_customers: List[int],
        current_route: List[int],
    ) -> Optional[int]:
        """
        Wybór kolejnego klienta na podstawie klasycznej reguły Ant System:
        p_ij ~ (tau_ij ^ alpha) * (eta_ij ^ beta)

        Uwzględniamy wyłącznie klientów, których dodanie do trasy
        nie narusza capacity i S_max.
        """
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
        """
        Jedna mrówka buduje pełne rozwiązanie:
        - dopóki są klienci,
        - tworzy kolejne trasy,
        - każda trasa startuje w depocie,
        - dodaje klientów probabilistycznie.
        """
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
        """
        Parowanie feromonu:
        tau = (1 - rho) * tau
        """
        self.pheromone_matrix *= (1.0 - self.config.evaporation)

    def _deposit_pheromones(self, routes: List[List[int]], total_length: float) -> None:
        """
        Odkładanie feromonu przez jedną mrówkę.
        Klasycznie:
        delta_tau = Q / L
        """
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

    def solve(self) -> AntSystemResult:
        best_routes: List[List[int]] = []
        best_length = float("inf")
        best_iteration = -1

        for iteration in range(self.config.iterations):
            iteration_solutions: List[Tuple[List[List[int]], float]] = []

            for _ in range(self.config.ants):
                routes = self._construct_solution()

                if not routes:
                    continue

                if not self.instance.is_solution_feasible(routes):
                    continue

                total_length = self.instance.solution_length(routes)
                iteration_solutions.append((routes, total_length))

                if total_length < best_length:
                    best_length = total_length
                    best_routes = routes
                    best_iteration = iteration + 1

            self._evaporate_pheromones()

            for routes, total_length in iteration_solutions:
                self._deposit_pheromones(routes, total_length)

        feasible = len(best_routes) > 0 and self.instance.is_solution_feasible(best_routes)

        return AntSystemResult(
            routes=best_routes,
            total_length=best_length if feasible else float("inf"),
            feasible=feasible,
            best_iteration=best_iteration,
        )