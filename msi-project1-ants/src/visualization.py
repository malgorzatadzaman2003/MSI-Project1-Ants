from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src.cvrp_instance import CVRPInstance


def ensure_results_dirs() -> None:
    """Tworzy katalogi wynikowe, jeśli jeszcze nie istnieją."""
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/routes").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)


def save_routes_text(
    instance: CVRPInstance,
    routes: List[List[int]],
    total_length: float,
    filepath: str,
) -> None:
    """Zapisuje rozwiązanie w czytelnej formie tekstowej."""
    lines = []
    lines.append(f"Instancja: {instance.name}")
    lines.append(f"Liczba pojazdów: {instance.num_vehicles}")
    lines.append(f"Capacity: {instance.capacity}")
    lines.append(f"S_max: {instance.s_max}")
    lines.append(f"Total length: {total_length:.2f}")
    lines.append("")

    for idx, route in enumerate(routes, start=1):
        load = instance.route_load(route)
        length = instance.route_length(route)
        route_str = " -> ".join(map(str, route))
        lines.append(f"Pojazd {idx}: depot -> {route_str} -> depot")
        lines.append(f"  Ładunek: {load}")
        lines.append(f"  Długość: {length:.2f}")
        lines.append("")

    Path(filepath).write_text("\n".join(lines), encoding="utf-8")


def save_routes_table(
    instance: CVRPInstance,
    routes: List[List[int]],
    total_length: float,
    execution_time: float,
    filepath: str,
) -> None:
    """Zapisuje tabelę z informacjami o trasach do CSV."""
    rows = []

    for idx, route in enumerate(routes, start=1):
        rows.append(
            {
                "instance": instance.name,
                "vehicle_id": idx,
                "route": "depot -> " + " -> ".join(map(str, route)) + " -> depot",
                "num_customers": len(route),
                "route_load": instance.route_load(route),
                "route_length": round(instance.route_length(route), 2),
                "capacity_limit": instance.capacity,
                "s_max": instance.s_max,
                "feasible": instance.is_route_feasible(route),
                "total_solution_length": round(total_length, 2),
                "execution_time_sec": execution_time,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)


def plot_routes(
    instance: CVRPInstance,
    routes: List[List[int]],
    filepath: str,
    title: str | None = None,
) -> None:
    """Rysuje trasy i zapisuje wykres do PNG."""
    ensure_results_dirs()

    fig, ax = plt.subplots(figsize=(10, 8))

    depot_x, depot_y = instance.node_coords[instance.depot_id]
    ax.scatter(depot_x, depot_y, s=180, marker="s", label="Depot")

    customers = instance.customers()
    customer_x = [instance.node_coords[c][0] for c in customers]
    customer_y = [instance.node_coords[c][1] for c in customers]

    ax.scatter(customer_x, customer_y, s=80)

    for c in customers:
        x, y = instance.node_coords[c]
        ax.text(x + 1.0, y + 1.0, f"{c}\n(d={instance.demands[c]})", fontsize=8)

    x_d, y_d = instance.node_coords[instance.depot_id]
    ax.text(x_d + 1.0, y_d + 1.0, "Depot", fontsize=10)

    for idx, route in enumerate(routes, start=1):
        full_route = [instance.depot_id] + route + [instance.depot_id]
        xs = [instance.node_coords[node][0] for node in full_route]
        ys = [instance.node_coords[node][1] for node in full_route]

        ax.plot(xs, ys, marker="o", linewidth=2, label=f"Pojazd {idx}")

    if title is None:
        title = f"CVRP routes - {instance.name}"

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close(fig)