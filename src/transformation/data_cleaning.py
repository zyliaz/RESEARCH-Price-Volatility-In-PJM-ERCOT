"""Data cleaning and preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess electricity market data."""

    def __init__(self):
        self.outlier_stats = {}

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_len = len(df)
        df = df.drop_duplicates(subset=subset)
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        method: str = "interpolate"
    ) -> pd.DataFrame:
        """
        Handle missing values in numeric columns.

        Args:
            df: Input DataFrame
            numeric_cols: List of numeric columns to process
            method: Method to handle missing values ('interpolate', 'ffill', 'drop')
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        missing_before = df[numeric_cols].isna().sum().sum()

        if method == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5)
        elif method == "ffill":
            df[numeric_cols] = df[numeric_cols].ffill(limit=5)
        elif method == "drop":
            df = df.dropna(subset=numeric_cols)

        missing_after = df[numeric_cols].isna().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")

        return df

    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        column: str,
        multiplier: float = 3.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Detect outliers using IQR method.

        Args:
            df: Input DataFrame
            column: Column to check for outliers
            multiplier: IQR multiplier (default 3.0 for extreme outliers)

        Returns:
            Tuple of (DataFrame with outlier flag, boolean Series of outliers)
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        self.outlier_stats[column] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_outliers": outliers.sum(),
            "pct_outliers": outliers.mean() * 100
        }

        logger.info(f"{column}: {outliers.sum()} outliers ({outliers.mean()*100:.2f}%)")

        return df, outliers

    def cap_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        lower_pct: float = 0.01,
        upper_pct: float = 0.99
    ) -> pd.DataFrame:
        """Cap outliers at specified percentiles."""
        lower = df[column].quantile(lower_pct)
        upper = df[column].quantile(upper_pct)

        original_min = df[column].min()
        original_max = df[column].max()

        df[column] = df[column].clip(lower=lower, upper=upper)

        logger.info(f"{column}: Capped from [{original_min:.2f}, {original_max:.2f}] to [{lower:.2f}, {upper:.2f}]")

        return df

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
        return df

    def add_datetime_features(self, df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
        """Add datetime-derived features."""
        if datetime_col not in df.columns:
            # Try to find a datetime column
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns
            if len(datetime_cols) > 0:
                datetime_col = datetime_cols[0]
            else:
                logger.warning("No datetime column found")
                return df

        df["hour"] = df[datetime_col].dt.hour
        df["day_of_week"] = df[datetime_col].dt.dayofweek
        df["month"] = df[datetime_col].dt.month
        df["year"] = df[datetime_col].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_peak_hour"] = df["hour"].between(7, 22).astype(int)

        # Season
        df["season"] = df["month"].map({
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall"
        })

        return df

    def clean_price_data(
        self,
        df: pd.DataFrame,
        price_col: str = "lmp",
        cap_negative: bool = True,
        cap_extreme: bool = True,
        extreme_threshold: float = 2000.0
    ) -> pd.DataFrame:
        """
        Clean price data with electricity market-specific logic.

        Args:
            df: Input DataFrame
            price_col: Name of price column
            cap_negative: Whether to cap negative prices at 0
            cap_extreme: Whether to cap extreme positive prices
            extreme_threshold: Threshold for extreme prices ($/MWh)
        """
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found")
            return df

        # Log original stats
        logger.info(f"Original {price_col} range: [{df[price_col].min():.2f}, {df[price_col].max():.2f}]")

        # Flag extreme values before cleaning
        df[f"{price_col}_spike"] = (df[price_col] > extreme_threshold).astype(int)
        df[f"{price_col}_negative"] = (df[price_col] < 0).astype(int)

        # Optionally cap values
        if cap_negative:
            n_negative = (df[price_col] < 0).sum()
            df[price_col] = df[price_col].clip(lower=0)
            if n_negative > 0:
                logger.info(f"Capped {n_negative} negative prices to 0")

        if cap_extreme:
            n_extreme = (df[price_col] > extreme_threshold).sum()
            df[price_col] = df[price_col].clip(upper=extreme_threshold)
            if n_extreme > 0:
                logger.info(f"Capped {n_extreme} extreme prices to {extreme_threshold}")

        return df
