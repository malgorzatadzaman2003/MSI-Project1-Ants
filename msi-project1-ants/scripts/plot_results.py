import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def prepare_instance_order(df: pd.DataFrame, instance_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["instance"] = pd.Categorical(df["instance"], categories=instance_order, ordered=True)
    return df.sort_values("instance")


def main() -> None:
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv("results/tables/benchmark_summary_ants10_iter1000_evap0.5.csv")
    raw_df = pd.read_csv("results/tables/benchmark_raw_ants10_iter1000_evap0.5.csv")

    instance_order = ["A-n32-k5", "A-n48-k7", "A-n60-k9", "A-n80-k10"]
    algorithm_order = ["greedy", "as", "acs", "mmas"]

    # Zachowaj tylko algorytmy, które rzeczywiście są w danych
    algorithm_order = [algo for algo in algorithm_order if algo in summary_df["algorithm"].unique()]

    # -----------------------------
    # Wykres 1: porównanie długości tras
    # -----------------------------
    plt.figure(figsize=(9, 5))

    for algo in algorithm_order:
        sub = summary_df[summary_df["algorithm"] == algo].copy()
        sub = prepare_instance_order(sub, instance_order)

        plt.plot(
            sub["instance"].astype(str),
            sub["mean_total_length"],
            marker="o",
            label=algo.upper(),
        )

    plt.title("Porównanie średniej długości tras (Ants=10, Iter=1000)")
    plt.xlabel("Instancja")
    plt.ylabel("Średnia długość tras")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/length_comparison_ants10_iter1000_evap0.5.png", dpi=200)
    plt.close()

    # -----------------------------
    # Wykres 2: porównanie czasu wykonania
    # -----------------------------
    plt.figure(figsize=(9, 5))

    for algo in algorithm_order:
        sub = summary_df[summary_df["algorithm"] == algo].copy()
        sub = prepare_instance_order(sub, instance_order)

        plt.plot(
            sub["instance"].astype(str),
            sub["mean_execution_time_sec"],
            marker="o",
            label=algo.upper(),
        )

    plt.title("Porównanie średniego czasu wykonania (Ants=10, Iter=1000)")
    plt.xlabel("Instancja")
    plt.ylabel("Czas [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/time_comparison_ants10_iter1000_evap0.5.png", dpi=200)
    plt.close()

    # -----------------------------
    # Wykres 3: boxplot stabilności wszystkich algorytmów losowych
    # -----------------------------
    stochastic_algorithms = [algo for algo in ["as", "acs", "mmas"] if algo in raw_df["algorithm"].unique()]

    plot_labels = []
    boxplot_data = []

    for instance in instance_order:
        for algo in stochastic_algorithms:
            values = raw_df[
                (raw_df["instance"] == instance) &
                (raw_df["algorithm"] == algo)
            ]["total_length"].tolist()

            if values:
                boxplot_data.append(values)
                plot_labels.append(f"{instance}\n{algo.upper()}")

    if boxplot_data:
        plt.figure(figsize=(14, 6))
        plt.boxplot(boxplot_data, labels=plot_labels)
        plt.title("Stabilność wyników algorytmów mrówkowych (Ants=10, Iter=1000)")
        plt.xlabel("Instancja i algorytm")
        plt.ylabel("Całkowita długość tras")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/plots/aco_stability_boxplot_ants10_iter1000_evap0.5.png", dpi=200)
        plt.close()

    # -----------------------------
    # Wykresy 4-6: osobny boxplot dla AS, ACS, MMAS
    # -----------------------------
    for algo in stochastic_algorithms:
        algo_df = raw_df[raw_df["algorithm"] == algo].copy()
        algo_df = prepare_instance_order(algo_df, instance_order)

        boxplot_data = []
        labels = []

        for instance in instance_order:
            values = algo_df[algo_df["instance"] == instance]["total_length"].tolist()
            if values:
                boxplot_data.append(values)
                labels.append(instance)

        if boxplot_data:
            plt.figure(figsize=(9, 5))
            plt.boxplot(boxplot_data, labels=labels)
            plt.title(f"Stabilność wyników {algo.upper()} (Ants=10, Iter=1000)")
            plt.xlabel("Instancja")
            plt.ylabel("Całkowita długość tras")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/plots/{algo}_stability_boxplot_ants10_iter1000_evap0.5.png", dpi=200)
            plt.close()

    # -----------------------------
    # Wykres 7: odchylenie standardowe długości tras
    # -----------------------------
    plt.figure(figsize=(9, 5))

    for algo in [a for a in ["as", "acs", "mmas"] if a in summary_df["algorithm"].unique()]:
        sub = summary_df[summary_df["algorithm"] == algo].copy()
        sub = prepare_instance_order(sub, instance_order)

        plt.plot(
            sub["instance"].astype(str),
            sub["std_total_length"],
            marker="o",
            label=algo.upper(),
        )

    plt.title("Odchylenie standardowe długości tras (Ants=10, Iter=1000)")
    plt.xlabel("Instancja")
    plt.ylabel("Odchylenie standardowe")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/std_comparison_ants10_iter1000_evap0.5.png", dpi=200)
    plt.close()

    print("Wykresy zapisane w results/plots/")


if __name__ == "__main__":
    main()