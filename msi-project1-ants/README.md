# MSI-Project1-Ants

Projekt porównuje algorytm zachłanny oraz warianty ACO (AS, ACS, MMAS) dla problemu CVRP z dodatkowym ograniczeniem maksymalnej długości trasy \(S_max\), z wykorzystaniem instancji CVRPLIB.

## Uruchomienie
W przypadku zmiany danych tj. ilości mrówek, iteracji, itd należy je zmienić w main() odpowiednio w plikach run_benchmark.py lub run_single.py 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m scripts.run_benchmark
python3 -m scripts.run_single

```

## Struktura projektu

- `src/` – implementacja algorytmów
- `scripts/` – skrypty uruchamiające eksperymenty
- `data/raw/` – instancje CVRP
- `results/tables/` – tabele z wynikami
- `plots/` – wygenerowane wykresy
- `report/` – raport końcowy

## Raport

Pełny opis problemu, hipotez, metodologii i wyników znajduje się w raporcie końcowym w pliku `MSI_Mrówki_Raport.pdf` umieszczonym na pltformie Leon w zadaniu na kanale przedmiotu.
