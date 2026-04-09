import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    Path("results/plots").mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv("results/tables/benchmark_summary.csv")
    raw_df = pd.read_csv("results/tables/benchmark_raw.csv")

    # Zachowaj kolejność instancji z benchmarku
    instance_order = ["A-n32-k5", "A-n48-k7", "A-n60-k9", "A-n80-k10"]

    # -----------------------------
    # Wykres 1: porównanie długości tras
    # -----------------------------
    plt.figure(figsize=(8, 5))

    for algo in summary_df["algorithm"].unique():
        sub = summary_df[summary_df["algorithm"] == algo].copy()
        sub["instance"] = pd.Categorical(sub["instance"], categories=instance_order, ordered=True)
        sub = sub.sort_values("instance")

        plt.plot(
            sub["instance"].astype(str),
            sub["mean_total_length"],
            marker="o",
            label=algo,
        )

    plt.title("Porównanie długości tras")
    plt.xlabel("Instancja")
    plt.ylabel("Średnia długość tras")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/length_comparison.png", dpi=200)
    plt.close()

    # -----------------------------
    # Wykres 2: porównanie czasu wykonania
    # -----------------------------
    plt.figure(figsize=(8, 5))

    for algo in summary_df["algorithm"].unique():
        sub = summary_df[summary_df["algorithm"] == algo].copy()
        sub["instance"] = pd.Categorical(sub["instance"], categories=instance_order, ordered=True)
        sub = sub.sort_values("instance")

        plt.plot(
            sub["instance"].astype(str),
            sub["mean_execution_time_sec"],
            marker="o",
            label=algo,
        )

    plt.title("Porównanie czasu wykonania")
    plt.xlabel("Instancja")
    plt.ylabel("Czas [s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/time_comparison.png", dpi=200)
    plt.close()

    # -----------------------------
    # Wykres 3: boxplot stabilności AS
    # -----------------------------
    as_df = raw_df[raw_df["algorithm"] == "as"].copy()
    as_df["instance"] = pd.Categorical(as_df["instance"], categories=instance_order, ordered=True)
    as_df = as_df.sort_values("instance")

    boxplot_data = []
    labels = []

    for instance in instance_order:
        values = as_df[as_df["instance"] == instance]["total_length"].tolist()
        if values:
            boxplot_data.append(values)
            labels.append(instance)

    plt.figure(figsize=(9, 5))
    plt.boxplot(boxplot_data, labels=labels)
    plt.title("Stabilność wyników Ant System")
    plt.xlabel("Instancja")
    plt.ylabel("Całkowita długość tras")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/as_stability_boxplot.png", dpi=200)
    plt.close()

    print("Wykresy zapisane w results/plots/")


if __name__ == "__main__":
    main()