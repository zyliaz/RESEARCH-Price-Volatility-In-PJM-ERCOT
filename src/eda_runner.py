"""
EDA Runner Script - Compatible with Python 3.9+

This script performs exploratory data analysis on ERCOT and PJM electricity market data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all data files."""
    print("=" * 70)
    print("Loading Data")
    print("=" * 70)

    data = {}

    # ERCOT data
    ercot_lmp = pd.read_parquet(DATA_RAW / "ercot" / "ercot_lmp.parquet")
    ercot_load = pd.read_parquet(DATA_RAW / "ercot" / "ercot_load.parquet")
    ercot_fuel = pd.read_parquet(DATA_RAW / "ercot" / "ercot_fuel_mix.parquet")

    print(f"ERCOT LMP: {ercot_lmp.shape}")
    print(f"ERCOT Load: {ercot_load.shape}")
    print(f"ERCOT Fuel Mix: {ercot_fuel.shape}")

    # PJM data
    pjm_lmp = pd.read_parquet(DATA_RAW / "pjm" / "pjm_lmp_zone.parquet")
    pjm_load = pd.read_parquet(DATA_RAW / "pjm" / "pjm_load.parquet")
    pjm_fuel = pd.read_parquet(DATA_RAW / "pjm" / "pjm_fuel_mix.parquet")

    print(f"PJM LMP: {pjm_lmp.shape}")
    print(f"PJM Load: {pjm_load.shape}")
    print(f"PJM Fuel Mix: {pjm_fuel.shape}")

    return {
        'ercot_lmp': ercot_lmp,
        'ercot_load': ercot_load,
        'ercot_fuel': ercot_fuel,
        'pjm_lmp': pjm_lmp,
        'pjm_load': pjm_load,
        'pjm_fuel': pjm_fuel
    }


def analyze_price_distribution(ercot_lmp, pjm_lmp):
    """Analyze price distributions."""
    print("\n" + "=" * 70)
    print("Price Distribution Analysis")
    print("=" * 70)

    # Find price column
    ercot_price_col = 'LMP' if 'LMP' in ercot_lmp.columns else 'lmp'
    pjm_price_col = 'LMP' if 'LMP' in pjm_lmp.columns else 'lmp'

    print(f"\nERCOT West Price Statistics ($/MWh):")
    print(ercot_lmp[ercot_price_col].describe())

    print(f"\nPJM DOM Price Statistics ($/MWh):")
    print(pjm_lmp[pjm_price_col].describe())

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ERCOT histogram
    ax1 = axes[0, 0]
    ercot_prices = ercot_lmp[ercot_price_col].dropna()
    ercot_prices_clipped = ercot_prices.clip(-50, 150)
    ax1.hist(ercot_prices_clipped, bins=80, alpha=0.7, color='steelblue', edgecolor='white')
    ax1.axvline(ercot_prices.median(), color='red', linestyle='--', linewidth=2, label=f'Median: ${ercot_prices.median():.2f}')
    ax1.axvline(ercot_prices.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: ${ercot_prices.mean():.2f}')
    ax1.set_xlabel('Price ($/MWh)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('ERCOT West - Price Distribution')
    ax1.legend()

    # PJM histogram
    ax2 = axes[0, 1]
    pjm_prices = pjm_lmp[pjm_price_col].dropna()
    pjm_prices_clipped = pjm_prices.clip(-50, 150)
    ax2.hist(pjm_prices_clipped, bins=80, alpha=0.7, color='forestgreen', edgecolor='white')
    ax2.axvline(pjm_prices.median(), color='red', linestyle='--', linewidth=2, label=f'Median: ${pjm_prices.median():.2f}')
    ax2.axvline(pjm_prices.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: ${pjm_prices.mean():.2f}')
    ax2.set_xlabel('Price ($/MWh)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('PJM DOM - Price Distribution')
    ax2.legend()

    # Box plots comparison
    ax3 = axes[1, 0]
    data_for_box = [ercot_prices_clipped.values, pjm_prices_clipped.values]
    bp = ax3.boxplot(data_for_box, labels=['ERCOT West', 'PJM DOM'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('forestgreen')
    ax3.set_ylabel('Price ($/MWh)')
    ax3.set_title('Price Distribution Comparison')

    # Log-scale histogram
    ax4 = axes[1, 1]
    ax4.hist(ercot_prices[ercot_prices > 0], bins=80, alpha=0.5, label='ERCOT West', color='steelblue')
    ax4.hist(pjm_prices[pjm_prices > 0], bins=80, alpha=0.5, label='PJM DOM', color='forestgreen')
    ax4.set_yscale('log')
    ax4.set_xlabel('Price ($/MWh)')
    ax4.set_ylabel('Frequency (log scale)')
    ax4.set_title('Price Distribution - Log Scale (Tail Analysis)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: price_distribution.png")

    return ercot_price_col, pjm_price_col


def calculate_volatility(df, price_col, time_col='Time'):
    """Calculate volatility metrics."""
    df = df.copy()
    df = df.sort_values(time_col)

    # Returns
    df['returns'] = df[price_col].pct_change()
    df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))

    # Rolling volatility
    df['rolling_vol_24h'] = df['returns'].rolling(window=24, min_periods=12).std()
    df['rolling_vol_168h'] = df['returns'].rolling(window=168, min_periods=84).std()

    # Price range
    df['rolling_range_24h'] = df[price_col].rolling(window=24).max() - df[price_col].rolling(window=24).min()

    # Squared returns
    df['squared_returns'] = df['returns'] ** 2

    return df


def analyze_volatility(ercot_lmp, pjm_lmp, ercot_price_col, pjm_price_col):
    """Analyze volatility patterns."""
    print("\n" + "=" * 70)
    print("Volatility Analysis")
    print("=" * 70)

    # Calculate volatility
    ercot_vol = calculate_volatility(ercot_lmp, ercot_price_col)
    pjm_vol = calculate_volatility(pjm_lmp, pjm_price_col)

    print(f"\nERCOT Volatility Statistics:")
    print(f"  Mean 24h Vol: {ercot_vol['rolling_vol_24h'].mean():.4f}")
    print(f"  Max 24h Vol: {ercot_vol['rolling_vol_24h'].max():.4f}")
    print(f"  Return Std: {ercot_vol['returns'].std():.4f}")
    print(f"  Return Skewness: {ercot_vol['returns'].skew():.4f}")
    print(f"  Return Kurtosis: {ercot_vol['returns'].kurtosis():.4f}")

    print(f"\nPJM Volatility Statistics:")
    print(f"  Mean 24h Vol: {pjm_vol['rolling_vol_24h'].mean():.4f}")
    print(f"  Max 24h Vol: {pjm_vol['rolling_vol_24h'].max():.4f}")
    print(f"  Return Std: {pjm_vol['returns'].std():.4f}")
    print(f"  Return Skewness: {pjm_vol['returns'].skew():.4f}")
    print(f"  Return Kurtosis: {pjm_vol['returns'].kurtosis():.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Returns distribution
    ax1 = axes[0, 0]
    ercot_returns = ercot_vol['returns'].dropna().clip(-1, 1)
    pjm_returns = pjm_vol['returns'].dropna().clip(-1, 1)
    ax1.hist(ercot_returns, bins=80, alpha=0.5, label='ERCOT West', color='steelblue', density=True)
    ax1.hist(pjm_returns, bins=80, alpha=0.5, label='PJM DOM', color='forestgreen', density=True)
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.set_title('Return Distribution Comparison')
    ax1.legend()

    # Rolling volatility time series
    ax2 = axes[0, 1]
    ax2.plot(ercot_vol['Time'], ercot_vol['rolling_vol_24h'], alpha=0.7, label='ERCOT West', color='steelblue', linewidth=0.8)
    ax2.plot(pjm_vol['Time'], pjm_vol['rolling_vol_24h'], alpha=0.7, label='PJM DOM', color='forestgreen', linewidth=0.8)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('24h Rolling Volatility')
    ax2.set_title('Rolling Volatility Over Time')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    # Volatility distribution
    ax3 = axes[1, 0]
    ax3.hist(ercot_vol['rolling_vol_24h'].dropna().clip(0, 0.5), bins=50, alpha=0.5, label='ERCOT West', color='steelblue')
    ax3.hist(pjm_vol['rolling_vol_24h'].dropna().clip(0, 0.5), bins=50, alpha=0.5, label='PJM DOM', color='forestgreen')
    ax3.set_xlabel('24h Rolling Volatility')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Volatility Distribution')
    ax3.legend()

    # Price range distribution
    ax4 = axes[1, 1]
    ax4.hist(ercot_vol['rolling_range_24h'].dropna().clip(0, 100), bins=50, alpha=0.5, label='ERCOT West', color='steelblue')
    ax4.hist(pjm_vol['rolling_range_24h'].dropna().clip(0, 100), bins=50, alpha=0.5, label='PJM DOM', color='forestgreen')
    ax4.set_xlabel('24h Price Range ($/MWh)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('24-Hour Price Range Distribution')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'volatility_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: volatility_analysis.png")

    return ercot_vol, pjm_vol


def analyze_temporal_patterns(ercot_vol, pjm_vol, ercot_price_col, pjm_price_col):
    """Analyze temporal patterns."""
    print("\n" + "=" * 70)
    print("Temporal Patterns Analysis")
    print("=" * 70)

    # Add temporal features
    for df in [ercot_vol, pjm_vol]:
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek
        df['month'] = df['Time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average price by hour
    ax1 = axes[0, 0]
    hourly_ercot = ercot_vol.groupby('hour')[ercot_price_col].mean()
    hourly_pjm = pjm_vol.groupby('hour')[pjm_price_col].mean()
    ax1.plot(hourly_ercot.index, hourly_ercot.values, 'o-', label='ERCOT West', color='steelblue', linewidth=2, markersize=6)
    ax1.plot(hourly_pjm.index, hourly_pjm.values, 's-', label='PJM DOM', color='forestgreen', linewidth=2, markersize=6)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Price ($/MWh)')
    ax1.set_title('Average Price by Hour of Day')
    ax1.legend()
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, alpha=0.3)

    # Volatility by hour
    ax2 = axes[0, 1]
    hourly_vol_ercot = ercot_vol.groupby('hour')['rolling_vol_24h'].mean()
    hourly_vol_pjm = pjm_vol.groupby('hour')['rolling_vol_24h'].mean()
    ax2.plot(hourly_vol_ercot.index, hourly_vol_ercot.values, 'o-', label='ERCOT West', color='steelblue', linewidth=2, markersize=6)
    ax2.plot(hourly_vol_pjm.index, hourly_vol_pjm.values, 's-', label='PJM DOM', color='forestgreen', linewidth=2, markersize=6)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average 24h Rolling Volatility')
    ax2.set_title('Volatility by Hour of Day')
    ax2.legend()
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3)

    # Average price by day of week
    ax3 = axes[1, 0]
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_ercot = ercot_vol.groupby('day_of_week')[ercot_price_col].mean()
    daily_pjm = pjm_vol.groupby('day_of_week')[pjm_price_col].mean()
    x = np.arange(7)
    width = 0.35
    ax3.bar(x - width/2, daily_ercot.values, width, label='ERCOT West', color='steelblue')
    ax3.bar(x + width/2, daily_pjm.values, width, label='PJM DOM', color='forestgreen')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Average Price ($/MWh)')
    ax3.set_title('Average Price by Day of Week')
    ax3.set_xticks(x)
    ax3.set_xticklabels(day_names)
    ax3.legend()

    # Weekend vs Weekday
    ax4 = axes[1, 1]
    categories = ['ERCOT\nWeekday', 'ERCOT\nWeekend', 'PJM\nWeekday', 'PJM\nWeekend']
    values = [
        ercot_vol[ercot_vol['is_weekend'] == 0][ercot_price_col].mean(),
        ercot_vol[ercot_vol['is_weekend'] == 1][ercot_price_col].mean(),
        pjm_vol[pjm_vol['is_weekend'] == 0][pjm_price_col].mean(),
        pjm_vol[pjm_vol['is_weekend'] == 1][pjm_price_col].mean()
    ]
    colors = ['steelblue', 'lightsteelblue', 'forestgreen', 'lightgreen']
    ax4.bar(categories, values, color=colors)
    ax4.set_ylabel('Average Price ($/MWh)')
    ax4.set_title('Weekday vs Weekend Price Comparison')

    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: temporal_patterns.png")

    return ercot_vol, pjm_vol


def analyze_fuel_mix(ercot_fuel, pjm_fuel):
    """Analyze generation fuel mix."""
    print("\n" + "=" * 70)
    print("Fuel Mix Analysis")
    print("=" * 70)

    # Calculate average generation
    fuel_cols_ercot = [c for c in ercot_fuel.columns if c != 'Time']
    fuel_cols_pjm = [c for c in pjm_fuel.columns if c != 'Time']

    ercot_avg = ercot_fuel[fuel_cols_ercot].mean()
    pjm_avg = pjm_fuel[fuel_cols_pjm].mean()

    print("\nERCOT Average Generation (MW):")
    for fuel, val in ercot_avg.sort_values(ascending=False).items():
        pct = val / ercot_avg.sum() * 100
        print(f"  {fuel}: {val:,.0f} MW ({pct:.1f}%)")

    print("\nPJM Average Generation (MW):")
    for fuel, val in pjm_avg.sort_values(ascending=False).items():
        pct = val / pjm_avg.sum() * 100
        print(f"  {fuel}: {val:,.0f} MW ({pct:.1f}%)")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ERCOT pie chart
    ax1 = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(ercot_avg)))
    wedges, texts, autotexts = ax1.pie(
        ercot_avg.values,
        labels=ercot_avg.index,
        autopct='%1.1f%%',
        colors=colors,
        pctdistance=0.8
    )
    ax1.set_title('ERCOT Generation Mix')

    # PJM pie chart
    ax2 = axes[1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(pjm_avg)))
    wedges, texts, autotexts = ax2.pie(
        pjm_avg.values,
        labels=pjm_avg.index,
        autopct='%1.1f%%',
        colors=colors,
        pctdistance=0.8
    )
    ax2.set_title('PJM Generation Mix')

    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'fuel_mix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: fuel_mix.png")


def analyze_price_spikes(ercot_vol, pjm_vol, ercot_price_col, pjm_price_col):
    """Analyze price spike patterns."""
    print("\n" + "=" * 70)
    print("Price Spike Analysis")
    print("=" * 70)

    # Identify spikes (> 3x rolling mean)
    for df, price_col, name in [(ercot_vol, ercot_price_col, 'ERCOT'), (pjm_vol, pjm_price_col, 'PJM')]:
        rolling_mean = df[price_col].rolling(window=168, min_periods=24).mean()
        df['spike_threshold'] = rolling_mean * 3
        df['is_spike'] = (df[price_col] > df['spike_threshold']).astype(int)
        df['is_extreme_spike'] = (df[price_col] > rolling_mean * 5).astype(int)

        spike_pct = df['is_spike'].mean() * 100
        extreme_pct = df['is_extreme_spike'].mean() * 100
        print(f"\n{name}:")
        print(f"  Spikes (>3x rolling mean): {df['is_spike'].sum()} ({spike_pct:.2f}%)")
        print(f"  Extreme Spikes (>5x): {df['is_extreme_spike'].sum()} ({extreme_pct:.2f}%)")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ERCOT spike by hour
    ax1 = axes[0]
    spike_by_hour = ercot_vol.groupby('hour')['is_spike'].mean() * 100
    ax1.bar(spike_by_hour.index, spike_by_hour.values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Spike Probability (%)')
    ax1.set_title('ERCOT West - Price Spike Probability by Hour')
    ax1.set_xticks(range(0, 24, 2))

    # PJM spike by hour
    ax2 = axes[1]
    spike_by_hour = pjm_vol.groupby('hour')['is_spike'].mean() * 100
    ax2.bar(spike_by_hour.index, spike_by_hour.values, color='forestgreen', alpha=0.7)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Spike Probability (%)')
    ax2.set_title('PJM DOM - Price Spike Probability by Hour')
    ax2.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(DATA_PROCESSED / 'spike_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: spike_analysis.png")

    return ercot_vol, pjm_vol


def create_summary(ercot_vol, pjm_vol, ercot_price_col, pjm_price_col):
    """Create summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY: ERCOT West vs PJM DOM Comparison")
    print("=" * 70)

    summary = []

    # Price metrics
    summary.append(('Mean Price ($/MWh)', f"{ercot_vol[ercot_price_col].mean():.2f}", f"{pjm_vol[pjm_price_col].mean():.2f}"))
    summary.append(('Median Price ($/MWh)', f"{ercot_vol[ercot_price_col].median():.2f}", f"{pjm_vol[pjm_price_col].median():.2f}"))
    summary.append(('Std Dev Price', f"{ercot_vol[ercot_price_col].std():.2f}", f"{pjm_vol[pjm_price_col].std():.2f}"))
    summary.append(('Max Price ($/MWh)', f"{ercot_vol[ercot_price_col].max():.2f}", f"{pjm_vol[pjm_price_col].max():.2f}"))
    summary.append(('Min Price ($/MWh)', f"{ercot_vol[ercot_price_col].min():.2f}", f"{pjm_vol[pjm_price_col].min():.2f}"))
    summary.append(('Price Skewness', f"{ercot_vol[ercot_price_col].skew():.4f}", f"{pjm_vol[pjm_price_col].skew():.4f}"))
    summary.append(('Price Kurtosis', f"{ercot_vol[ercot_price_col].kurtosis():.4f}", f"{pjm_vol[pjm_price_col].kurtosis():.4f}"))

    # Volatility metrics
    summary.append(('Mean 24h Volatility', f"{ercot_vol['rolling_vol_24h'].mean():.4f}", f"{pjm_vol['rolling_vol_24h'].mean():.4f}"))
    summary.append(('Max 24h Volatility', f"{ercot_vol['rolling_vol_24h'].max():.4f}", f"{pjm_vol['rolling_vol_24h'].max():.4f}"))
    summary.append(('Mean 24h Range ($/MWh)', f"{ercot_vol['rolling_range_24h'].mean():.2f}", f"{pjm_vol['rolling_range_24h'].mean():.2f}"))

    # Spike metrics
    summary.append(('Spike Rate (>3x, %)', f"{ercot_vol['is_spike'].mean()*100:.2f}", f"{pjm_vol['is_spike'].mean()*100:.2f}"))
    summary.append(('Extreme Spike Rate (>5x, %)', f"{ercot_vol['is_extreme_spike'].mean()*100:.2f}", f"{pjm_vol['is_extreme_spike'].mean()*100:.2f}"))

    # Print summary table
    print("\n{:<30} {:>15} {:>15}".format('Metric', 'ERCOT West', 'PJM DOM'))
    print("-" * 60)
    for metric, ercot_val, pjm_val in summary:
        print("{:<30} {:>15} {:>15}".format(metric, ercot_val, pjm_val))

    # Save summary to CSV
    summary_df = pd.DataFrame(summary, columns=['Metric', 'ERCOT West', 'PJM DOM'])
    summary_df.to_csv(DATA_PROCESSED / 'eda_summary.csv', index=False)
    print("\nSaved: eda_summary.csv")

    return summary_df


def save_processed_data(ercot_vol, pjm_vol):
    """Save processed data."""
    print("\n" + "=" * 70)
    print("Saving Processed Data")
    print("=" * 70)

    ercot_vol.to_parquet(DATA_PROCESSED / 'ercot_west_processed.parquet', index=False)
    pjm_vol.to_parquet(DATA_PROCESSED / 'pjm_dom_processed.parquet', index=False)

    print(f"Saved ERCOT processed data: {len(ercot_vol)} rows")
    print(f"Saved PJM processed data: {len(pjm_vol)} rows")


def main():
    """Main EDA pipeline."""
    print("\n" + "=" * 70)
    print("ERCOT vs PJM Price Volatility - Exploratory Data Analysis")
    print("=" * 70)

    # Load data
    data = load_data()

    # Analyze price distributions
    ercot_price_col, pjm_price_col = analyze_price_distribution(
        data['ercot_lmp'], data['pjm_lmp']
    )

    # Analyze volatility
    ercot_vol, pjm_vol = analyze_volatility(
        data['ercot_lmp'], data['pjm_lmp'],
        ercot_price_col, pjm_price_col
    )

    # Analyze temporal patterns
    ercot_vol, pjm_vol = analyze_temporal_patterns(
        ercot_vol, pjm_vol,
        ercot_price_col, pjm_price_col
    )

    # Analyze fuel mix
    analyze_fuel_mix(data['ercot_fuel'], data['pjm_fuel'])

    # Analyze price spikes
    ercot_vol, pjm_vol = analyze_price_spikes(
        ercot_vol, pjm_vol,
        ercot_price_col, pjm_price_col
    )

    # Create summary
    summary_df = create_summary(ercot_vol, pjm_vol, ercot_price_col, pjm_price_col)

    # Save processed data
    save_processed_data(ercot_vol, pjm_vol)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)
    print("\nGenerated files in data/processed/:")
    print("  - ercot_west_processed.parquet")
    print("  - pjm_dom_processed.parquet")
    print("  - eda_summary.csv")
    print("  - price_distribution.png")
    print("  - volatility_analysis.png")
    print("  - temporal_patterns.png")
    print("  - fuel_mix.png")
    print("  - spike_analysis.png")


if __name__ == "__main__":
    main()
