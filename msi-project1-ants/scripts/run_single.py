from pathlib import Path

from src.cvrplib_parser import parse_cvrplib
from src.greedy import solve_greedy


def print_solution(instance_name: str, routes: list[list[int]], total_length: float, feasible: bool) -> None:
    print("=" * 60)
    print(f"Instancja: {instance_name}")
    print(f"Rozwiązanie dopuszczalne: {feasible}")
    print(f"Całkowita długość tras: {total_length:.2f}")
    print("-" * 60)

    for idx, route in enumerate(routes, start=1):
        route_str = " -> ".join(map(str, route))
        print(f"Pojazd {idx}: depot -> {route_str} -> depot")

    print("=" * 60)


def main() -> None:
    data_path = Path("data/raw/A-n32-k5.vrp")

    instance = parse_cvrplib(
        filepath=str(data_path),
        num_vehicles=5,
        s_max=500.0,
    )

    result = solve_greedy(instance)

    print_solution(
        instance_name=instance.name,
        routes=result.routes,
        total_length=result.total_length,
        feasible=result.feasible,
    )


if __name__ == "__main__":
    main()