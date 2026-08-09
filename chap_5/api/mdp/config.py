import os
from pathlib import Path
from dotenv import load_dotenv

CORE_DATA_DIR = Path(__file__).resolve().parent
ROOT_DIR = CORE_DATA_DIR.parents[2]
load_dotenv(ROOT_DIR / ".env")

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "master")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BOOTSTRAP = os.getenv("BOOTSTRAP", "true").lower() in ("true", "1", "t")

K = int(os.getenv("MDP_K", 3))
FRACTION = float(os.getenv("MDP_FRACTION", 0.001))
MAX_SKIP = int(os.getenv("MDP_MAX_SKIP", 5))
BATCH_SIZE = int(os.getenv("MDP_BATCH_SIZE", 1_000_000))

GAMMA_BOOST = float(os.getenv("MDP_GAMMA_BOOST", 1 / 1000))
GAMMA_RL = float(os.getenv("MDP_GAMMA_RL", 0.9))
LIST_SIZE = int(os.getenv("MDP_LIST_SIZE", 3))
THRESHOLD = float(os.getenv("MDP_THRESHOLD", 1e-3))
BOLTZMANN_TEMPERATURE = float(os.getenv("MDP_TEMPERATURE", 1.0))

DATA_DIR = CORE_DATA_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
NPY_DIR = DATA_DIR / "npy"

FLAT_PATH = NPY_DIR / "sequences_flat.npy"
OFFSETS_PATH = NPY_DIR / "sequences_offsets.npy"
USER_IDS_PATH = NPY_DIR / "user_ids.csv"
FULL_HIST_PATH = CLEANED_DIR / "historique_full.csv"

