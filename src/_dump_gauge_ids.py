import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
hydrodl = script_dir / "hydroDLpack"
if hydrodl.is_dir():
    sys.path.insert(0, str(hydrodl))
sys.path.insert(0, str(script_dir))

from config import ExperimentConfig
from data_utils import ensure_gauge_ids

cfg = ExperimentConfig()
ids = ensure_gauge_ids(cfg)
print(f"n={len(ids)}")
print(f"first={ids[0]} last={ids[-1]}")
print(f"path={cfg.gauge_id_path()}")
