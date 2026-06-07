from __future__ import annotations
from pathlib import Path
import zipfile
import requests
import pandas as pd

DATASET_SLUG = "martj42/international-football-results-from-1872-to-2017"
KAGGLE_VIEW_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_SLUG}"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"

ALIASES = {
    "United States": "United States", "USA": "United States", "US": "United States",
    "Turkey": "Türkiye", "Türkiye": "Türkiye",
    "Czech Republic": "Czechia", "Czechia": "Czechia",
    "South Korea": "South Korea", "Korea Republic": "South Korea",
    "DR Congo": "DR Congo", "Congo DR": "DR Congo", "Democratic Republic of the Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast", "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cape Verde", "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao", "Curaçao": "Curaçao",
    "Netherlands": "Netherlands", "Holland": "Netherlands",
}

def canonical_team(name: str) -> str:
    return ALIASES.get(str(name).strip(), str(name).strip())

def kaggle_metadata() -> dict:
    r = requests.get(KAGGLE_VIEW_URL, headers={"User-Agent":"HermesAgent/1.0"}, timeout=30)
    r.raise_for_status(); return r.json()

def download_kaggle(out_dir: str | Path = "data/raw") -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "international-football-results.zip"
    r = requests.get(KAGGLE_DOWNLOAD_URL, headers={"User-Agent":"HermesAgent/1.0"}, timeout=120)
    r.raise_for_status(); zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    return out / "results.csv"

def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team"] = df["home_team"].map(canonical_team)
    df["away_team"] = df["away_team"].map(canonical_team)
    return df
