"""PJM data extraction using gridstatus library."""

import pandas as pd
import gridstatus
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional, List

from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PJMExtractor:
    """Extract data from PJM using gridstatus."""

    def __init__(self):
        self.pjm = gridstatus.PJM()
        self.output_dir = config.RAW_DATA_DIR / "pjm"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_lmp_data(
        self,
        start_date: str,
        end_date: str,
        market: str = "REAL_TIME_HOURLY",
        location_type: str = "ZONE"
    ) -> pd.DataFrame:
        """
        Extract LMP data.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            market: Market type (REAL_TIME_HOURLY, DAY_AHEAD_HOURLY, REAL_TIME_5_MIN)
            location_type: Location type (ZONE, HUB, GEN, LOAD, etc.)

        Returns:
            DataFrame with LMP data
        """
        logger.info(f"Extracting PJM LMP data from {start_date} to {end_date}")

        all_data = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Extract in monthly chunks
        while current_date < end:
            chunk_end = min(current_date + timedelta(days=30), end)
            try:
                df = self.pjm.get_lmp(
                    start=current_date.strftime("%b %d, %Y"),
                    end=chunk_end.strftime("%b %d, %Y"),
                    market=market,
                    location_type=location_type
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  Extracted {len(df)} rows for {current_date.strftime('%Y-%m')}")
            except Exception as e:
                logger.warning(f"  Error extracting {current_date}: {e}")

            current_date = chunk_end

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total PJM LMP rows: {len(result)}")
            return result
        return pd.DataFrame()

    def extract_load_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Extract load data.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            DataFrame with load data
        """
        logger.info(f"Extracting PJM load data from {start_date} to {end_date}")

        all_data = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date < end:
            chunk_end = min(current_date + timedelta(days=30), end)
            try:
                df = self.pjm.get_load(
                    start=current_date.strftime("%b %d, %Y"),
                    end=chunk_end.strftime("%b %d, %Y")
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  Extracted {len(df)} rows for {current_date.strftime('%Y-%m')}")
            except Exception as e:
                logger.warning(f"  Error extracting {current_date}: {e}")

            current_date = chunk_end

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total PJM load rows: {len(result)}")
            return result
        return pd.DataFrame()

    def extract_fuel_mix(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Extract generation fuel mix data.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            DataFrame with fuel mix data
        """
        logger.info(f"Extracting PJM fuel mix data from {start_date} to {end_date}")

        all_data = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date < end:
            chunk_end = min(current_date + timedelta(days=30), end)
            try:
                df = self.pjm.get_fuel_mix(
                    start=current_date.strftime("%b %d, %Y"),
                    end=chunk_end.strftime("%b %d, %Y")
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  Extracted {len(df)} rows for {current_date.strftime('%Y-%m')}")
            except Exception as e:
                logger.warning(f"  Error extracting {current_date}: {e}")

            current_date = chunk_end

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total PJM fuel mix rows: {len(result)}")
            return result
        return pd.DataFrame()

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to parquet file."""
        filepath = self.output_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")

    def run_full_extraction(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """Run full extraction pipeline."""
        start = start_date or config.START_DATE
        end = end_date or config.END_DATE

        # Extract LMP data (zone-level)
        lmp_df = self.extract_lmp_data(start, end, location_type="ZONE")
        if len(lmp_df) > 0:
            self.save_to_parquet(lmp_df, "pjm_lmp_zone.parquet")

        # Extract load data
        load_df = self.extract_load_data(start, end)
        if len(load_df) > 0:
            self.save_to_parquet(load_df, "pjm_load.parquet")

        # Extract fuel mix
        fuel_df = self.extract_fuel_mix(start, end)
        if len(fuel_df) > 0:
            self.save_to_parquet(fuel_df, "pjm_fuel_mix.parquet")

        return {
            "lmp": lmp_df,
            "load": load_df,
            "fuel_mix": fuel_df
        }


if __name__ == "__main__":
    extractor = PJMExtractor()
    # For testing, extract just 1 month
    extractor.run_full_extraction("2024-01-01", "2024-01-31")
