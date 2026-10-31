"""Generate a synthetic telecom churn CSV (stdlib only) so the harness runs out of the box."""
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from ..config import Settings


def generate(path: Path, rows: int = 5000, seed: int = 7) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
                    "PaymentMethod", "SeniorCitizen", "TechSupport", "Churn"])
        for i in range(rows):
            tenure = rng.randint(0, 72)
            contract = rng.choices(["Month-to-month", "One year", "Two year"], [0.55, 0.25, 0.20])[0]
            internet = rng.choices(["Fiber optic", "DSL", "No"], [0.45, 0.35, 0.20])[0]
            monthly = round({"Fiber optic": 80, "DSL": 55, "No": 25}[internet] + rng.gauss(0, 10), 2)
            total = round(monthly * tenure, 2)
            senior = int(rng.random() < 0.16)
            tech = rng.choice(["Yes", "No"]) if internet != "No" else "No internet service"
            pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            z = (-0.8 - 0.045 * tenure + 0.012 * (monthly - 60)
                 + {"Month-to-month": 1.0, "One year": -0.2, "Two year": -1.0}[contract]
                 + (0.4 if pay == "Electronic check" else 0) + 0.3 * senior - (0.4 if tech == "Yes" else 0))
            churn = "Yes" if rng.random() < 1 / (1 + math.exp(-z)) else "No"
            total_str = " " if tenure == 0 else f"{total}"  # realistic dirt: blank numeric strings
            w.writerow([f"C{i:05d}", tenure, monthly, total_str, contract, internet, pay, senior, tech, churn])
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()
    out = generate(Settings.from_env().paths.raw / "churn.csv", rows=args.rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
