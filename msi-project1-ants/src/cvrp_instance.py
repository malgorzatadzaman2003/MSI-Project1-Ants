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