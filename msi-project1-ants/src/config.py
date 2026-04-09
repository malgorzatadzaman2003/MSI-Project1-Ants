from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    seed: int = 42
    ants: int = 20
    iterations: int = 100
    alpha: float = 1.0
    beta: float = 3.0
    evaporation: float = 0.5
    q: float = 100.0


@dataclass
class ExperimentConfig:
    repeats: int = 30
    save_routes: bool = True
    save_tables: bool = True