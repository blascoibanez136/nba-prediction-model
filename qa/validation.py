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
