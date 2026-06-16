"""Project configuration and filesystem paths."""

from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
OUTPUTS_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

load_dotenv(REPO_ROOT / ".env")


def ensure_directories() -> None:
    """Create expected local data and output directories if they are missing."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        TABLES_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
