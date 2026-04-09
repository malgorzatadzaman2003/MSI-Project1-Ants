from pathlib import Path
from time import perf_counter

from src.cvrplib_parser import parse_cvrplib
from src.greedy import solve_greedy

from src.visualization import (
    ensure_results_dirs,
    plot_routes,
    save_routes_table,
    save_routes_text,
)


def print_solution_summary(instance, result, execution_time: float) -> None:
    print("=" * 72)
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

    instance = parse_cvrplib(
        filepath=str(data_path),
        num_vehicles=5,
        s_max=500.0,
    )

    start = perf_counter()
    result = solve_greedy(instance)
    end = perf_counter()

    execution_time = end - start

    print_solution_summary(instance, result, execution_time)

    base_name = f"{instance.name}_greedy"

    routes_txt_path = Path("results/routes") / f"{base_name}.txt"
    routes_csv_path = Path("results/tables") / f"{base_name}.csv"
    plot_path = Path("results/plots") / f"{base_name}.png"

    save_routes_text(
        instance=instance,
        routes=result.routes,
        total_length=result.total_length,
        filepath=str(routes_txt_path),
    )

    save_routes_table(
        instance=instance,
        routes=result.routes,
        total_length=result.total_length,
        execution_time=execution_time,
        filepath=str(routes_csv_path),
    )

    plot_routes(
        instance=instance,
        routes=result.routes,
        filepath=str(plot_path),
        title=f"Greedy solution - {instance.name}",
    )

    print("Zapisano wyniki:")
    print(f"  TXT:  {routes_txt_path}")
    print(f"  CSV:  {routes_csv_path}")
    print(f"  PNG:  {plot_path}")


if __name__ == "__main__":
    main()