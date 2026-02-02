"""
Direct API access for ERCOT and PJM data.
Compatible with Python 3.9+

This module provides fallback data extraction when gridstatus is not available.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERCOTDirectAPI:
    """
    Direct access to ERCOT public data.

    ERCOT provides public data through their data portal and direct file downloads.
    """

    BASE_URL = "https://www.ercot.com/api/1/services/read/dashboards"
    REPORT_URL = "https://www.ercot.com/misapp/GetReports.do"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Research/1.0)',
            'Accept': 'application/json'
        })

    def get_current_fuel_mix(self) -> pd.DataFrame:
        """Get current fuel mix data."""
        url = f"{self.BASE_URL}/fuel-mix.json"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'data' in data:
                df = pd.DataFrame(data['data'])
                return df
        except Exception as e:
            logger.error(f"Error fetching ERCOT fuel mix: {e}")
        return pd.DataFrame()

    def get_current_load(self) -> pd.DataFrame:
        """Get current system load data."""
        url = f"{self.BASE_URL}/loadForecastVsActual.json"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'data' in data:
                df = pd.DataFrame(data['data'])
                return df
        except Exception as e:
            logger.error(f"Error fetching ERCOT load: {e}")
        return pd.DataFrame()

    def get_current_prices(self) -> pd.DataFrame:
        """Get current SPP data."""
        url = f"{self.BASE_URL}/real-time-spp.json"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'data' in data:
                df = pd.DataFrame(data['data'])
                return df
        except Exception as e:
            logger.error(f"Error fetching ERCOT prices: {e}")
        return pd.DataFrame()


class PJMDirectAPI:
    """
    Direct access to PJM Data Miner 2 API.

    Note: For full historical data access, you may need a PJM account.
    This provides access to publicly available data.
    """

    BASE_URL = "https://api.pjm.com/api/v1"
    DATAMINER_URL = "https://dataminer2.pjm.com/feed"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('PJM_API_KEY', '')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Ocp-Apim-Subscription-Key': self.api_key
            })
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Research/1.0)',
            'Accept': 'application/json'
        })

    def get_lmp_data(
        self,
        start_date: str,
        end_date: str,
        zone: str = "DOM"
    ) -> pd.DataFrame:
        """
        Get LMP data from PJM Data Miner.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            zone: Zone filter (e.g., "DOM", "PECO")
        """
        url = f"{self.DATAMINER_URL}/rt_hrl_lmps/definition"

        try:
            # First get feed definition
            response = self.session.get(url, timeout=30)
            logger.info(f"PJM API response status: {response.status_code}")

            # For public access, we may need to use alternative methods
            # This is a placeholder for actual API implementation
            logger.warning("PJM API requires authentication for historical data")

        except Exception as e:
            logger.error(f"Error fetching PJM data: {e}")

        return pd.DataFrame()


def create_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Create sample data for development when API access is limited.

    This generates synthetic data that mimics the structure of real data
    for development and testing purposes.
    """
    import numpy as np

    logger.info("Generating sample data for development...")

    # Date range: 3 months of hourly data
    start = datetime(2024, 10, 1)
    end = datetime(2024, 12, 31)
    hours = pd.date_range(start=start, end=end, freq='H')

    np.random.seed(42)

    # ERCOT West sample data
    n = len(hours)

    # Base price with seasonality and noise
    hour_effect = np.sin(np.arange(n) * 2 * np.pi / 24) * 15  # Daily pattern
    week_effect = np.sin(np.arange(n) * 2 * np.pi / (24 * 7)) * 5  # Weekly pattern
    base_price = 35 + hour_effect + week_effect

    # Add volatility clustering (GARCH-like)
    noise = np.random.randn(n)
    vol = np.zeros(n)
    vol[0] = 5
    for i in range(1, n):
        vol[i] = 2 + 0.1 * noise[i-1]**2 + 0.85 * vol[i-1]

    ercot_prices = base_price + np.sqrt(vol) * noise

    # Add occasional spikes
    spike_mask = np.random.random(n) < 0.01  # 1% spike probability
    ercot_prices[spike_mask] *= np.random.uniform(2, 5, spike_mask.sum())

    # ERCOT load (MW)
    base_load = 45000 + hour_effect * 1000 + week_effect * 500
    ercot_load = base_load + np.random.randn(n) * 2000

    ercot_df = pd.DataFrame({
        'Time': hours,
        'Location': 'LZ_WEST',
        'Location Type': 'Load Zone',
        'LMP': ercot_prices,
        'Energy': ercot_prices * 0.9,
        'Congestion': ercot_prices * 0.05,
        'Loss': ercot_prices * 0.05
    })

    # PJM DOM sample data
    # PJM typically has lower volatility due to capacity market
    pjm_base_price = 40 + hour_effect * 0.8 + week_effect * 0.6
    pjm_noise = np.random.randn(n)
    pjm_vol = np.zeros(n)
    pjm_vol[0] = 3
    for i in range(1, n):
        pjm_vol[i] = 1.5 + 0.08 * pjm_noise[i-1]**2 + 0.88 * pjm_vol[i-1]

    pjm_prices = pjm_base_price + np.sqrt(pjm_vol) * pjm_noise

    # Fewer spikes in PJM
    spike_mask = np.random.random(n) < 0.005  # 0.5% spike probability
    pjm_prices[spike_mask] *= np.random.uniform(1.5, 3, spike_mask.sum())

    pjm_df = pd.DataFrame({
        'Time': hours,
        'Location Name': 'DOM',
        'Location Type': 'ZONE',
        'LMP': pjm_prices,
        'Energy': pjm_prices * 0.85,
        'Congestion': pjm_prices * 0.08,
        'Loss': pjm_prices * 0.07
    })

    # Load data
    ercot_load_df = pd.DataFrame({
        'Time': hours,
        'ERCOT': ercot_load,
        'WEST': ercot_load * 0.15,
        'NORTH': ercot_load * 0.35,
        'SOUTH': ercot_load * 0.25,
        'HOUSTON': ercot_load * 0.25
    })

    pjm_base_load = 85000 + hour_effect * 2000 + week_effect * 1000
    pjm_load = pjm_base_load + np.random.randn(n) * 3000

    pjm_load_df = pd.DataFrame({
        'Time': hours,
        'Load': pjm_load,
        'Zone': 'DOM'
    })

    # Fuel mix data
    fuel_types_ercot = ['Gas', 'Wind', 'Coal', 'Nuclear', 'Solar', 'Hydro', 'Other']
    fuel_types_pjm = ['Gas', 'Nuclear', 'Coal', 'Wind', 'Hydro', 'Solar', 'Other']

    ercot_fuel = pd.DataFrame({'Time': hours})
    ercot_fuel['Gas'] = 35000 + np.random.randn(n) * 2000
    ercot_fuel['Wind'] = 15000 + np.sin(np.arange(n) * 2 * np.pi / 24) * 5000 + np.random.randn(n) * 3000
    ercot_fuel['Coal'] = 8000 + np.random.randn(n) * 500
    ercot_fuel['Nuclear'] = 5000 + np.random.randn(n) * 100
    ercot_fuel['Solar'] = np.maximum(0, 3000 * np.sin(np.arange(n) * 2 * np.pi / 24 - np.pi/4) + np.random.randn(n) * 500)
    ercot_fuel['Hydro'] = 500 + np.random.randn(n) * 50
    ercot_fuel['Other'] = 1000 + np.random.randn(n) * 100

    pjm_fuel = pd.DataFrame({'Time': hours})
    pjm_fuel['Gas'] = 45000 + np.random.randn(n) * 3000
    pjm_fuel['Nuclear'] = 30000 + np.random.randn(n) * 500
    pjm_fuel['Coal'] = 15000 + np.random.randn(n) * 1000
    pjm_fuel['Wind'] = 5000 + np.sin(np.arange(n) * 2 * np.pi / 24) * 2000 + np.random.randn(n) * 1000
    pjm_fuel['Hydro'] = 3000 + np.random.randn(n) * 200
    pjm_fuel['Solar'] = np.maximum(0, 2000 * np.sin(np.arange(n) * 2 * np.pi / 24 - np.pi/4) + np.random.randn(n) * 300)
    pjm_fuel['Other'] = 2000 + np.random.randn(n) * 200

    logger.info(f"Generated {n} hours of sample data")

    return {
        'ercot_lmp': ercot_df,
        'pjm_lmp': pjm_df,
        'ercot_load': ercot_load_df,
        'pjm_load': pjm_load_df,
        'ercot_fuel': ercot_fuel,
        'pjm_fuel': pjm_fuel
    }


def save_sample_data(output_dir: str = None):
    """Generate and save sample data to parquet files."""
    from pathlib import Path

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ercot").mkdir(exist_ok=True)
    (output_dir / "pjm").mkdir(exist_ok=True)

    data = create_sample_data()

    # Save ERCOT data
    data['ercot_lmp'].to_parquet(output_dir / "ercot" / "ercot_lmp.parquet", index=False)
    data['ercot_load'].to_parquet(output_dir / "ercot" / "ercot_load.parquet", index=False)
    data['ercot_fuel'].to_parquet(output_dir / "ercot" / "ercot_fuel_mix.parquet", index=False)

    # Save PJM data
    data['pjm_lmp'].to_parquet(output_dir / "pjm" / "pjm_lmp_zone.parquet", index=False)
    data['pjm_load'].to_parquet(output_dir / "pjm" / "pjm_load.parquet", index=False)
    data['pjm_fuel'].to_parquet(output_dir / "pjm" / "pjm_fuel_mix.parquet", index=False)

    logger.info(f"Saved sample data to {output_dir}")

    return data


if __name__ == "__main__":
    # Generate and save sample data for development
    save_sample_data()
