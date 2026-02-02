"""Temporal alignment and resampling utilities."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalAligner:
    """Align and resample time series data to common frequency."""

    def __init__(self, target_freq: str = "h"):
        """
        Initialize aligner.

        Args:
            target_freq: Target frequency ('h' for hourly, '15min', '5min', 'D' for daily)
        """
        self.target_freq = target_freq

    def to_utc(self, df: pd.DataFrame, datetime_col: str, source_tz: str) -> pd.DataFrame:
        """
        Convert datetime column to UTC.

        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            source_tz: Source timezone (e.g., 'US/Central' for ERCOT, 'US/Eastern' for PJM)
        """
        if datetime_col not in df.columns:
            logger.warning(f"Datetime column '{datetime_col}' not found")
            return df

        # Localize if naive, then convert to UTC
        if df[datetime_col].dt.tz is None:
            df[datetime_col] = df[datetime_col].dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")

        df[datetime_col] = df[datetime_col].dt.tz_convert("UTC")
        df["datetime_utc"] = df[datetime_col]

        return df

    def resample_to_hourly(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        value_cols: Optional[list] = None,
        agg_funcs: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample data to hourly frequency.

        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            value_cols: Columns to aggregate
            agg_funcs: Aggregation functions per column (default: mean)
        """
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if agg_funcs is None:
            agg_funcs = {col: "mean" for col in value_cols}

        df = df.set_index(datetime_col)
        df_hourly = df[value_cols].resample("h").agg(agg_funcs)
        df_hourly = df_hourly.reset_index()

        logger.info(f"Resampled from {len(df)} to {len(df_hourly)} rows (hourly)")

        return df_hourly

    def align_datasets(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        datetime_col: str = "datetime_utc",
        how: str = "inner"
    ) -> pd.DataFrame:
        """
        Align two datasets on datetime.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            datetime_col: Common datetime column
            how: Join type ('inner', 'outer', 'left', 'right')
        """
        merged = pd.merge(df1, df2, on=datetime_col, how=how, suffixes=("", "_y"))

        # Remove duplicate columns
        cols_to_drop = [c for c in merged.columns if c.endswith("_y")]
        merged = merged.drop(columns=cols_to_drop)

        logger.info(f"Aligned datasets: {len(df1)} + {len(df2)} -> {len(merged)} rows")

        return merged

    def fill_missing_hours(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fill missing hours with NaN values to create continuous time series.

        Args:
            df: Input DataFrame
            datetime_col: Datetime column
            start_date: Start of date range (optional)
            end_date: End of date range (optional)
        """
        if start_date is None:
            start_date = df[datetime_col].min()
        if end_date is None:
            end_date = df[datetime_col].max()

        # Create complete date range
        full_range = pd.date_range(start=start_date, end=end_date, freq="h")

        # Reindex
        df = df.set_index(datetime_col)
        df = df.reindex(full_range)
        df = df.reset_index()
        df = df.rename(columns={"index": datetime_col})

        missing = df.isna().any(axis=1).sum()
        if missing > 0:
            logger.info(f"Added {missing} missing hours")

        return df

    def calculate_intra_period_stats(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        value_col: str,
        source_freq: str = "15min"
    ) -> pd.DataFrame:
        """
        Calculate intra-period statistics (useful for volatility from sub-hourly data).

        Args:
            df: Input DataFrame at sub-hourly frequency
            datetime_col: Datetime column
            value_col: Value column (e.g., price)
            source_freq: Source frequency

        Returns:
            Hourly DataFrame with additional stats (std, min, max, range)
        """
        df = df.set_index(datetime_col)

        # Calculate hourly stats
        hourly_stats = df[value_col].resample("h").agg([
            "mean",
            "std",
            "min",
            "max",
            "count"
        ])

        hourly_stats["range"] = hourly_stats["max"] - hourly_stats["min"]

        # Calculate returns for volatility
        df["returns"] = df[value_col].pct_change()
        hourly_vol = df["returns"].resample("h").std()
        hourly_stats["realized_vol"] = hourly_vol

        hourly_stats = hourly_stats.reset_index()
        hourly_stats.columns = [datetime_col, f"{value_col}_mean", f"{value_col}_std",
                                f"{value_col}_min", f"{value_col}_max", f"{value_col}_count",
                                f"{value_col}_range", f"{value_col}_realized_vol"]

        return hourly_stats
