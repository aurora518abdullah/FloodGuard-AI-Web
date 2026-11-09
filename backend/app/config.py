from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_PATHS = [DATA_DIR / "all_stations.parquet", DATA_DIR / "all_stations.csv"]
MODEL_PATH = MODEL_DIR / "floodguard_models.joblib"
TARGETS = {
    "24h": "flood_within_24h",
    "48h": "flood_within_48h",
    "72h": "flood_within_72h",
}
