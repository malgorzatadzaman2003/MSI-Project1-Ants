from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

@dataclass
class CVRPInstance:     
    name: str
    dimension: int
    capacity: int
    depot_id: int
    node_coords: Dict[int, Tuple[float, float]]
    demands: Dict[int, int]
    num_vehicles: int
    s_max: float

    def customers(self) -> List[int]:
        """Zwraca listę klientów bez magazynu."""
        return [node for node in self.node_coords if node != self.depot_id]
    
    def distance(self, a: int, b: int) -> float:
        """Odległość euklidesowa między dwoma wierzchołkami/punktami."""
        x1, y1 = self.node_coords[a]
        x2, y2 = self.node_coords[b]
        return math.dist((x1, y1), (x2, y2))
    
    def route_length(self, route: List[int]) -> float:
        """
        Liczy długość pojedynczej trasy.
        Zakładamy, że route zawiera klientów, np. [1, 5, 3].
        Trasa liczona jest jako depot -> ... -> depot.
        """
        if not route:
            return 0.0

        total = 0.0
        prev = self.depot_id
        for customer in route:
            total += self.distance(prev, customer)
            prev = customer
        total += self.distance(prev, self.depot_id)
        return total
    
    def route_load(self, route: List[int]) -> int:
        """Suma zapotrzebowań klientów na trasie."""
        return sum(self.demands[c] for c in route)
    
    def is_route_feasible(self, route: List[int]) -> bool:
        """Sprawdza ograniczenie pojemności i S_max dla jednej trasy."""
        return (
            self.route_load(route) <= self.capacity
            and self.route_length(route) <= self.s_max
        )
    
    def is_solution_feasible(self, routes: List[List[int]]) -> bool:
        """
        Sprawdza:
        - czy każdy klient został odwiedzony dokładnie raz,
        - czy liczba tras <= liczba pojazdów,
        - czy każda trasa spełnia capacity i s_max.
        """
        if len(routes) > self.num_vehicles:
            return False

        visited = []
        for route in routes:
            if not self.is_route_feasible(route):
                return False
            visited.extend(route)

        return sorted(visited) == sorted(self.customers())
