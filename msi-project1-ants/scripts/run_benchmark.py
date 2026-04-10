from dataclasses import asdict

import pandas as pd

from src.experiment_runner import (
    run_as_once,
    run_acs_once,
    run_greedy_once,
    run_mmas_once,
    save_benchmark_outputs,
    summarize_results,
)


def main() -> None:
    # Instancje zgodne z konspektem
    instances = [
        {"path": "data/raw/A-n32-k5.vrp", "vehicles": 5, "s_max": 500.0},
        {"path": "data/raw/A-n48-k7.vrp", "vehicles": 7, "s_max": 600.0},
        {"path": "data/raw/A-n60-k9.vrp", "vehicles": 9, "s_max": 700.0},
        {"path": "data/raw/A-n80-k10.vrp", "vehicles": 10, "s_max": 900.0},
    ]

    # Parametry Ant System
    ants = 80
    iterations = 100
    alpha = 1.0
    beta = 3.0
    evaporation = 0.5
    q = 100.0

    # Liczba uruchomień AS
    seeds = list(range(30))

    all_results = []

    for spec in instances:
        print("=" * 72)
        print(f"Instancja: {spec['path']}")

        # Greedy - jedno uruchomienie wystarczy, bo jest deterministyczny
        greedy_result = run_greedy_once(
            instance_path=spec["path"],
            num_vehicles=spec["vehicles"],
            s_max=spec["s_max"],
        )
        all_results.append(asdict(greedy_result))
        print(
            f"[GREEDY] length={greedy_result.total_length:.2f}, "
            f"time={greedy_result.execution_time_sec:.6f}s"
        )

        # Ant System - wiele uruchomień z różnymi seedami
        for seed in seeds:
            as_result = run_as_once(
                instance_path=spec["path"],
                num_vehicles=spec["vehicles"],
                s_max=spec["s_max"],
                seed=seed,
                ants=ants,
                iterations=iterations,
                alpha=alpha,
                beta=beta,
                evaporation=evaporation,
                q=q,
            )
            all_results.append(asdict(as_result))

        print(f"[AS] wykonano {len(seeds)} uruchomień")

        # ACS
        for seed in seeds:
            acs_result = run_acs_once(
                instance_path=spec["path"],
                num_vehicles=spec["vehicles"],
                s_max=spec["s_max"],
                seed=seed,
                ants=ants,
                iterations=iterations,
                alpha=alpha,
                beta=beta,
                evaporation=evaporation,
                q=q,
            )
            all_results.append(asdict(acs_result))
        print(f"[ACS] wykonano {len(seeds)} uruchomień")

        # MMAS
        for seed in seeds:
            mmas_result = run_mmas_once(
                instance_path=spec["path"],
                num_vehicles=spec["vehicles"],
                s_max=spec["s_max"],
                seed=seed,
                ants=ants,
                iterations=iterations,
                alpha=alpha,
                beta=beta,
                evaporation=evaporation,
                q=q,
            )
            all_results.append(asdict(mmas_result))
        print(f"[MMAS] wykonano {len(seeds)} uruchomień")


    raw_df = pd.DataFrame(all_results)
    summary_df = summarize_results(raw_df)

    save_benchmark_outputs(
    raw_df=raw_df,
    summary_df=summary_df,
    ants=ants,
    iterations=iterations,
    )

    print("\nPodsumowanie:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()