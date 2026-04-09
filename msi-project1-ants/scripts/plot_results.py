import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/tables/benchmark_summary.csv")

# --- wykres długości ---
plt.figure()
for algo in df["algorithm"].unique():
    sub = df[df["algorithm"] == algo]
    plt.plot(sub["instance"], sub["mean_total_length"], marker="o", label=algo)

plt.title("Porównanie długości tras")
plt.xlabel("Instancja")
plt.ylabel("Średnia długość tras")
plt.legend()
plt.grid()
plt.savefig("results/plots/length_comparison.png")

# --- wykres czasu ---
plt.figure()
for algo in df["algorithm"].unique():
    sub = df[df["algorithm"] == algo]
    plt.plot(sub["instance"], sub["mean_execution_time_sec"], marker="o", label=algo)

plt.title("Porównanie czasu wykonania")
plt.xlabel("Instancja")
plt.ylabel("Czas [s]")
plt.legend()
plt.grid()
plt.savefig("results/plots/time_comparison.png")

print("Wykresy zapisane!")