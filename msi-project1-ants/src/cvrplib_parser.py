from pathlib import Path
from typing import Dict, Tuple
from src.cvrp_instance import CVRPInstance


def parse_cvrplib(filepath: str, num_vehicles: int, s_max: float) -> CVRPInstance:
    """
    Minimalny parser instancji .vrp z CVRPLIB.
    Obsługuje:
    - NAME
    - DIMENSION
    - CAPACITY
    - NODE_COORD_SECTION
    - DEMAND_SECTION
    - DEPOT_SECTION
    """
    path = Path(filepath)
    lines = path.read_text(encoding="utf-8").splitlines()

    name = path.stem
    dimension = None
    capacity = None
    node_coords: Dict[int, Tuple[float, float]] = {}
    demands: Dict[int, int] = {}
    depot_id = None

    mode = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("NAME"):
            name = line.split(":")[-1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[-1].strip())
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[-1].strip())
        elif line == "NODE_COORD_SECTION":
            mode = "coords"
        elif line == "DEMAND_SECTION":
            mode = "demands"
        elif line == "DEPOT_SECTION":
            mode = "depot"
        elif line == "EOF":
            break
        else:
            if mode == "coords":
                parts = line.split()
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords[node_id] = (x, y)
            elif mode == "demands":
                parts = line.split()
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand
            elif mode == "depot":
                value = int(line)
                if value != -1:
                    depot_id = value

    if dimension is None or capacity is None or depot_id is None:
        raise ValueError(f"Nie udało się poprawnie sparsować pliku {filepath}")

    return CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        depot_id=depot_id,
        node_coords=node_coords,
        demands=demands,
        num_vehicles=num_vehicles,
        s_max=s_max,
    )