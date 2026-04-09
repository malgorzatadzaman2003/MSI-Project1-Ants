from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ant_system import AntSystemSolver
from src.config import AlgorithmConfig
from src.cvrplib_parser import parse_cvrplib
from src.greedy import solve_greedy
from src.visualization import ensure_results_dirs


@dataclass
class SingleRunResult:
    algorithm: str
    instance: str
    seed: int
    feasible: bool
    total_length: float
    execution_time_sec: float
    num_routes: int
    best_iteration: Optional[int] = None


def run_greedy_once(instance_path: str, num_vehicles: int, s_max: float) -> SingleRunResult:
    instance = parse_cvrplib(
        filepath=instance_path,
        num_vehicles=num_vehicles,
        s_max=s_max,
    )

    start = perf_counter()
    result = solve_greedy(instance)
    end = perf_counter()

    return SingleRunResult(
        algorithm="greedy",
        instance=instance.name,
        seed=-1,
        feasible=result.feasible,
        total_length=result.total_length,
        execution_time_sec=end - start,
        num_routes=len(result.routes),
        best_iteration=None,
    )


def run_as_once(
    instance_path: str,
    num_vehicles: int,
    s_max: float,
    seed: int,
    ants: int,
    iterations: int,
    alpha: float,
    beta: float,
    evaporation: float,
    q: float,
) -> SingleRunResult:
    instance = parse_cvrplib(
        filepath=instance_path,
        num_vehicles=num_vehicles,
        s_max=s_max,
    )

    config = AlgorithmConfig(
        seed=seed,
        ants=ants,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
        evaporation=evaporation,
        q=q,
    )

    solver = AntSystemSolver(instance, config)

    start = perf_counter()
    result = solver.solve()
    end = perf_counter()

    return SingleRunResult(
        algorithm="as",
        instance=instance.name,
        seed=seed,
        feasible=result.feasible,
        total_length=result.total_length,
        execution_time_sec=end - start,
        num_routes=len(result.routes),
        best_iteration=result.best_iteration,
    )


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy tabelę zbiorczą:
    - średnia długość,
    - min, max,
    - odchylenie standardowe,
    - średni czas.
    """
    grouped = df.groupby(["algorithm", "instance"], as_index=False)

    rows: List[Dict[str, Any]] = []

    for (algorithm, instance), group in grouped:
        lengths = group["total_length"].tolist()
        times = group["execution_time_sec"].tolist()

        rows.append(
            {
                "algorithm": algorithm,
                "instance": instance,
                "runs": len(group),
                "feasible_rate": group["feasible"].mean(),
                "mean_total_length": mean(lengths),
                "min_total_length": min(lengths),
                "max_total_length": max(lengths),
                "std_total_length": stdev(lengths) if len(lengths) > 1 else 0.0,
                "mean_execution_time_sec": mean(times),
                "mean_num_routes": group["num_routes"].mean(),
            }
        )

    return pd.DataFrame(rows)


def save_benchmark_outputs(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    raw_filename: str = "benchmark_raw.csv",
    summary_filename: str = "benchmark_summary.csv",
) -> None:
    ensure_results_dirs()

    raw_path = Path("results/tables") / raw_filename
    summary_path = Path("results/tables") / summary_filename

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Zapisano benchmark:")
    print(f"  RAW:     {raw_path}")
    print(f"  SUMMARY: {summary_path}")