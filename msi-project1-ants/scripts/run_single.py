from pathlib import Path
from time import perf_counter

from src.cvrplib_parser import parse_cvrplib
from src.greedy import solve_greedy

from src.ant_system import AntSystemSolver
from src.config import AlgorithmConfig

from src.visualization import (
    ensure_results_dirs,
    plot_routes,
    save_routes_table,
    save_routes_text,
)


def print_solution_summary(instance, result, execution_time: float, algorithm_name: str) -> None:
    print("=" * 72)
    print(f"Algorytm:                 {algorithm_name}")
    print(f"Instancja:                {instance.name}")
    print(f"Liczba pojazdów:          {instance.num_vehicles}")
    print(f"Pojemność pojazdu:        {instance.capacity}")
    print(f"Maks. długość trasy:      {instance.s_max}")
    print(f"Rozwiązanie dopuszczalne: {result.feasible}")
    print(f"Całkowita długość tras:   {result.total_length:.2f}")
    print(f"Czas wykonania [s]:       {execution_time:.6f}")
    print("=" * 72)

    for idx, route in enumerate(result.routes, start=1):
        load = instance.route_load(route)
        length = instance.route_length(route)
        route_str = " -> ".join(map(str, route))

        print(f"Pojazd {idx}")
        print(f"  Trasa:    depot -> {route_str} -> depot")
        print(f"  Klienci:  {len(route)}")
        print(f"  Ładunek:  {load}/{instance.capacity}")
        print(f"  Długość:  {length:.2f}")
        print("-" * 72)


def main() -> None:
    ensure_results_dirs()

    data_path = Path("data/raw/A-n32-k5.vrp")
    algorithm = "as"   # "greedy" albo "as" na razie

    instance = parse_cvrplib(
        filepath=str(data_path),
        num_vehicles=5,
        s_max=500.0,
    )

    start = perf_counter()

    if algorithm == "greedy":
        result = solve_greedy(instance)
        routes = result.routes
        total_length = result.total_length
        feasible = result.feasible
        algorithm_name = "Greedy"

    elif algorithm == "as":
        config = AlgorithmConfig(
            seed=42,
            ants=20,
            iterations=100,
            alpha=1.0,
            beta=3.0,
            evaporation=0.5,
            q=100.0,
        )
        solver = AntSystemSolver(instance, config)
        result = solver.solve()
        routes = result.routes
        total_length = result.total_length
        feasible = result.feasible
        algorithm_name = f"Ant System (best iter: {result.best_iteration})"

    else:
        raise ValueError(f"Nieznany algorytm: {algorithm}")

    end = perf_counter()
    execution_time = end - start

    print_solution_summary(
    instance=instance,
    result=result,
    execution_time=execution_time,
    algorithm_name=algorithm_name,
    )

    base_name = f"{instance.name}_{algorithm}"

    routes_txt_path = Path("results/routes") / f"{base_name}.txt"
    routes_csv_path = Path("results/tables") / f"{base_name}.csv"
    plot_path = Path("results/plots") / f"{base_name}.png"

    save_routes_text(
        instance=instance,
        routes=routes,
        total_length=total_length,
        filepath=str(routes_txt_path),
    )

    save_routes_table(
        instance=instance,
        routes=routes,
        total_length=total_length,
        execution_time=execution_time,
        filepath=str(routes_csv_path),
    )

    plot_routes(
        instance=instance,
        routes=routes,
        filepath=str(plot_path),
        title=f"{algorithm_name} - {instance.name}",
    )

    print("Zapisano wyniki:")
    print(f"  TXT:  {routes_txt_path}")
    print(f"  CSV:  {routes_csv_path}")
    print(f"  PNG:  {plot_path}")


if __name__ == "__main__":
    main()