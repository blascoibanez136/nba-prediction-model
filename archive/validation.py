import sys
from pathlib import Path

# Add project root to sys.path so "src" package can be found
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import yaml
import importlib

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

required = ["project_name", "paths", "data_sources", "model"]
for key in required:
    assert key in cfg, f"Missing {key} in config/config.yaml"

# import checks
importlib.import_module("src.ingest.nba_ingest")
importlib.import_module("src.ingest.odds_ingest")
importlib.import_module("src.features.build_features")
importlib.import_module("src.model.train_model")

print("QA OK")
