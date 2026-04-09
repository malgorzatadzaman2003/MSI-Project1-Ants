from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.cvrp_instance import CVRPInstance


@dataclass
class GreedyResult:
    routes: List[List[int]]
    total_length: float
    feasible: bool


def _find_best_next_customer(
    instance: CVRPInstance,
    current_node: int,
    remaining_customers: List[int],
    current_route: List[int],
) -> Optional[int]:
    """
    Wybiera najbliższego klienta, którego można jeszcze dodać do bieżącej trasy
    bez naruszenia ograniczenia pojemności i S_max.

    Zwraca:
    - identyfikator klienta, jeśli znaleziono poprawny ruch,
    - None, jeśli nie da się już nikogo dodać do tej trasy.
    """
    feasible_candidates: List[Tuple[float, int]] = []

    for customer in remaining_customers:
        candidate_route = current_route + [customer]

        if instance.is_route_feasible(candidate_route):
            distance = instance.distance(current_node, customer)
            feasible_candidates.append((distance, customer))

    if not feasible_candidates:
        return None

    feasible_candidates.sort(key=lambda x: x[0])
    return feasible_candidates[0][1]


def solve_greedy(instance: CVRPInstance) -> GreedyResult:
    """
    Rozwiązuje CVRP heurystyką zachłanną typu nearest feasible neighbor.

    Strategia:
    1. Startujemy z pustą listą tras.
    2. Dopóki istnieją nieobsłużeni klienci:
       - zaczynamy nową trasę z magazynu,
       - dokładamy najbliższego klienta, którego da się dodać
         bez złamania capacity i S_max,
       - gdy nie da się dodać nikogo więcej, kończymy trasę.
    3. Jeśli liczba utworzonych tras przekroczy liczbę pojazdów, 
       odrzucamy rozwiązanie.
    """
    remaining_customers = sorted(instance.customers())
    routes: List[List[int]] = []

    while remaining_customers:
        if len(routes) >= instance.num_vehicles:
            return GreedyResult(
                routes=routes,
                total_length=float("inf"),
                feasible=False,
            )

        current_route: List[int] = []
        current_node = instance.depot_id

        while True:
            next_customer = _find_best_next_customer(
                instance=instance,
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
            return GreedyResult(
                routes=routes,
                total_length=float("inf"),
                feasible=False,
            )

        routes.append(current_route)

    feasible = instance.is_solution_feasible(routes)
    total_length = instance.solution_length(routes) if feasible else float("inf")

    return GreedyResult(
        routes=routes,
        total_length=total_length,
        feasible=feasible,
    )