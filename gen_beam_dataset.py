#!/usr/bin/env python3
# gen_beam_dataset.py

import numpy as np
import pandas as pd

def generate_beam_dataset(
    n_samples: int = 200,
    seed: int = 42,
    save_path: str = "beam_dataset.csv"
):
    rng = np.random.default_rng(seed)

    # 범위는 예시일 뿐이므로, 필요 시 엔지니어링 관점에서 조정 가능
    # E: 70 ~ 210 GPa
    E = rng.uniform(70e9, 210e9, size=n_samples)
    # I: 1e-8 ~ 5e-6 m^4
    I = rng.uniform(1e-8, 5e-6, size=n_samples)
    # L: 0.5 ~ 3.0 m
    L = rng.uniform(0.5, 3.0, size=n_samples)
    # P: 100 ~ 3000 N
    P = rng.uniform(100.0, 3000.0, size=n_samples)

    # Cantilever end deflection: delta = P L^3 / (3 E I)
    y_max = P * L**3 / (3.0 * E * I)

    df = pd.DataFrame(
        {
            "E": E,
            "I": I,
            "L": L,
            "P": P,
            "y_max": y_max,
        }
    )

    df.to_csv(save_path, index=False)
    print(f"Saved dataset to {save_path}")
    print(df.head())

if __name__ == "__main__":
    # 필요 시 n_samples, seed, save_path를 조정
    generate_beam_dataset(
        n_samples=200,
        seed=42,
        save_path="beam_dataset.csv",
    )
