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
    
    