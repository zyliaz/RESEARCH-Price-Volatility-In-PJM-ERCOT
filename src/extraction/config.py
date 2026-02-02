"""Configuration settings for data extraction."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Config:
    """Configuration for data extraction pipeline."""

    # Date range
    START_DATE: str = "2022-01-01"
    END_DATE: str = "2025-01-31"

    # ERCOT settings
    ERCOT_ZONES: list = None
    ERCOT_FOCUS_ZONE: str = "WEST"

    # PJM settings
    PJM_ZONES: list = None
    PJM_FOCUS_ZONE: str = "DOM"

    # Data paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    RAW_DATA_DIR: Path = None
    PROCESSED_DATA_DIR: Path = None

    def __post_init__(self):
        if self.ERCOT_ZONES is None:
            self.ERCOT_ZONES = ["WEST", "NORTH", "SOUTH", "HOUSTON"]
        if self.PJM_ZONES is None:
            self.PJM_ZONES = ["DOM", "PECO", "PEPCO", "BGE"]
        if self.RAW_DATA_DIR is None:
            self.RAW_DATA_DIR = self.PROJECT_ROOT / "data" / "raw"
        if self.PROCESSED_DATA_DIR is None:
            self.PROCESSED_DATA_DIR = self.PROJECT_ROOT / "data" / "processed"

        # Create directories if they don't exist
        (self.RAW_DATA_DIR / "ercot").mkdir(parents=True, exist_ok=True)
        (self.RAW_DATA_DIR / "pjm").mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = Config()
