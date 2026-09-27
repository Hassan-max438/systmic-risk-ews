"""Central configuration for the Systemic Risk Early Warning System."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

for p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_ASSETS = 8
N_DAYS = 5000
CRISIS_PROB = 0.02
CRISIS_PERSIST = 0.97
VOL_NORMAL = 0.01
VOL_CRISIS = 0.04
CORR_NORMAL = 0.20
CORR_CRISIS = 0.85

HORIZON = 10
WINDOWS = [30, 60, 90]
TEST_FRACTION = 0.20

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    random_state=SEED,
    n_jobs=-1,
)
MODEL_PATH = MODELS_DIR / "systemic_risk_ews.pkl"