
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

DEMO_FILE = ROOT / "data" / "demo_district_nutrition.csv"
NFHS_FILE = ROOT / "data" / "NFHS5_district_nutrition_cleaned.csv"

INDICATORS = {
    "SH.STA.STNT.ZS": "Stunting prevalence, children under 5 (%)",
    "SH.STA.WAST.ZS": "Wasting prevalence, children under 5 (%)",
    "SH.STA.MALN.ZS": "Underweight prevalence, children under 5 (%)",
}


def fetch_world_bank_nutrition():
    """Fetch historical public nutrition indicators from the World Bank API."""
    rows = []

    for code, label in INDICATORS.items():
        url = (
            f"https://api.worldbank.org/v2/country/IND/indicator/"
            f"{code}?format=json&per_page=100"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            continue

        for item in payload[1]:
            if item.get("value") is not None:
                rows.append({
                    "year": int(item["date"]),
                    "indicator": label,
                    "value": float(item["value"]),
                    "source": "World Bank Indicators API",
                })

    return pd.DataFrame(
        rows, columns=["year", "indicator", "value", "source"]
    )


def load_demo_dataset():
    """Load the original synthetic demonstration dataset."""
    return pd.read_csv(DEMO_FILE)


def load_nfhs_dataset():
    """Load NFHS-5 district survey data and standardize dashboard columns."""
    if not NFHS_FILE.exists():
        raise FileNotFoundError(
            f"NFHS-5 CSV not found: {NFHS_FILE}"
        )

    df = pd.read_csv(NFHS_FILE)

    column_mapping = {
        "stunted_under5_pct": "stunting_pct",
        "wasted_under5_pct": "wasting_pct",
        "underweight_under5_pct": "underweight_pct",
    }

    df = df.rename(columns=column_mapping)

    required = [
        "district",
        "state_ut",
        "stunting_pct",
        "wasting_pct",
        "underweight_pct",
        "households_surveyed",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from NFHS-5 CSV: {missing}"
        )

    df = df.rename(columns={"state_ut": "state"})

    numeric_columns = [
        "stunting_pct",
        "wasting_pct",
        "underweight_pct",
        "households_surveyed",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["district", "state"])

    return df
