import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.config import AlgorithmConfig
from src.cvrp_instance import CVRPInstance


@dataclass
class ACSResult:
    routes: List[List[int]]
    total_length: float
    feasible: bool
    best_iteration: int


class AntColonySystemSolver:
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

        # tau0 - początkowy poziom feromonu
        self.tau0 = 1.0
        self.pheromone_matrix = self._initialize_pheromones()

        # Parametr ACS:
        # q0 - prawdopodobieństwo wyboru najlepszego ruchu zamiast losowania
        self.q0 = 0.7

        # Parametr lokalnej aktualizacji
        self.local_evaporation = 0.05

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
        Heurystyka:
        eta_ij = 1 / distance(i, j)
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
        n = len(self.nodes)
        return np.full((n, n), self.tau0, dtype=float)

    def _feasible_customers(
        self,
        remaining_customers: List[int],
        current_route: List[int],
    ) -> List[int]:
        feasible = []

        for customer in remaining_customers:
            candidate_route = current_route + [customer]
            if self.instance.is_route_feasible(candidate_route):
                feasible.append(customer)

        return feasible

    def _state_transition_rule(
        self,
        current_node: int,
        feasible_customers: List[int],
    ) -> Optional[int]:
        """
        Reguła przejścia ACS:
        - z prawdopodobieństwem q0 wybieramy najlepszy ruch (eksploatacja),
        - w przeciwnym wypadku losujemy wg rozkładu zależnego od feromonu i heurystyki (eksploracja).
        """
        if not feasible_customers:
            return None

        current_idx = self.node_to_idx[current_node]

        attractiveness = []
        for customer in feasible_customers:
            customer_idx = self.node_to_idx[customer]
            tau = self.pheromone_matrix[current_idx, customer_idx]
            eta = self.heuristic_matrix[current_idx, customer_idx]
            value = (tau ** self.config.alpha) * (eta ** self.config.beta)
            attractiveness.append((customer, value))

        q = self.rng.random()

        # Eksploatacja: wybór najlepszego ruchu
        if q <= self.q0:
            best_customer = max(attractiveness, key=lambda x: x[1])[0]
            return best_customer

        # Eksploracja: losowanie probabilistyczne
        weights = [value for _, value in attractiveness]
        total_weight = sum(weights)

        if total_weight == 0:
            return self.rng.choice(feasible_customers)

        customers = [customer for customer, _ in attractiveness]
        probabilities = [w / total_weight for w in weights]

        return self.rng.choices(customers, weights=probabilities, k=1)[0]

    def _local_pheromone_update(self, a: int, b: int) -> None:
        """
        Lokalna aktualizacja ACS:
        tau_ij = (1 - xi) * tau_ij + xi * tau0
        """
        a_idx = self.node_to_idx[a]
        b_idx = self.node_to_idx[b]

        xi = self.local_evaporation

        self.pheromone_matrix[a_idx, b_idx] = (
            (1.0 - xi) * self.pheromone_matrix[a_idx, b_idx] + xi * self.tau0
        )
        self.pheromone_matrix[b_idx, a_idx] = self.pheromone_matrix[a_idx, b_idx]

    def _construct_solution(self) -> List[List[int]]:
        """
        Jedna mrówka buduje pełne rozwiązanie.
        Po każdym przejściu wykonujemy lokalną aktualizację feromonu.
        """
        remaining_customers = self.customers.copy()
        routes: List[List[int]] = []

        while remaining_customers:
            if len(routes) >= self.instance.num_vehicles:
                return []

            current_route: List[int] = []
            current_node = self.instance.depot_id

            while True:
                feasible_customers = self._feasible_customers(
                    remaining_customers=remaining_customers,
                    current_route=current_route,
                )

                next_customer = self._state_transition_rule(
                    current_node=current_node,
                    feasible_customers=feasible_customers,
                )

                if next_customer is None:
                    break

                # lokalna aktualizacja na przejściu current_node -> next_customer
                self._local_pheromone_update(current_node, next_customer)

                current_route.append(next_customer)
                remaining_customers.remove(next_customer)
                current_node = next_customer

            if not current_route:
                return []

            # lokalna aktualizacja powrotu do depotu
            self._local_pheromone_update(current_node, self.instance.depot_id)

            routes.append(current_route)

        return routes

    def _global_evaporation(self) -> None:
        self.pheromone_matrix *= (1.0 - self.config.evaporation)

    def _global_pheromone_update(self, routes: List[List[int]], total_length: float) -> None:
        """
        W ACS globalnie wzmacniamy tylko najlepsze rozwiązanie.
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

    def solve(self) -> ACSResult:
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

            self._global_evaporation()

            if best_routes:
                self._global_pheromone_update(best_routes, best_length)
            
        feasible = len(best_routes) > 0 and self.instance.is_solution_feasible(best_routes)

        return ACSResult(
            routes=best_routes,
            total_length=best_length if feasible else float("inf"),
            feasible=feasible,
            best_iteration=best_iteration,
        )