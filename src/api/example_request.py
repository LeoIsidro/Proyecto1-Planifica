import json
from pathlib import Path

import joblib
import pandas as pd
import requests

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
API_URL = "http://localhost:8000"


def build_payload_from_test_set(n: int = 5) -> list[dict]:
    test_df = pd.read_parquet(ARTIFACTS_DIR / "test.parquet")
    pp = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    feature_cols = pp["feature_cols"]

    sample = test_df[feature_cols + ["TransactionID"]].head(n)

    payloads = []
    for _, row in sample.iterrows():
        features = {
            col: (
                None
                if pd.isna(row[col])
                else (
                    int(row[col])
                    if isinstance(row[col], (int,))
                    else float(row[col]) if isinstance(row[col], (float,)) else str(row[col])
                )
            )
            for col in feature_cols
        }
        payloads.append(
            {
                "transaction_id": int(row["TransactionID"]),
                "features": features,
            }
        )
    return payloads


def main() -> None:
    print("Healthcheck:", requests.get(f"{API_URL}/health").json())
    print("Version:   ", requests.get(f"{API_URL}/version").json())

    payloads = build_payload_from_test_set(n=5)

    print("\n--- /predict (1 transacción) ---")
    r = requests.post(f"{API_URL}/predict", json=payloads[0])
    print(json.dumps(r.json(), indent=2))

    print("\n--- /predict/batch (5 transacciones) ---")
    r = requests.post(
        f"{API_URL}/predict/batch",
        json={"transactions": payloads},
    )
    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    main()
