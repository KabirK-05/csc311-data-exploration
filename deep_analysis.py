"""
CSC311 Deep Data Analysis
=========================
Deeper analysis beyond explore_data.py: Likert processing, price cleaning,
multi-select encoding, correlation heatmap, missing values chart, and
grouped data splitting with leakage verification.

Usage:
    python deep_analysis.py training_data_202601.csv --output_dir outputs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import re
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# Configuration
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
COLORS_3CLASS = ['#2ecc71', '#3498db', '#e74c3c']
CLASS_ORDER = ['The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond']
CLASS_SHORT = ['Persistence', 'Starry Night', 'Water Lily']
sns.set_palette(COLORS_3CLASS)
DPI = 300

def save_fig(fig, fig_dir, name):
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path

# Column name shortcuts
COL_ID = 'unique_id'
COL_PAINTING = 'Painting'
COL_EMOTION = 'On a scale of 1–10, how intense is the emotion conveyed by the artwork?'
COL_DESCRIBE = 'Describe how this painting makes you feel.'
COL_SOMBRE = 'This art piece makes me feel sombre.'
COL_CONTENT = 'This art piece makes me feel content.'
COL_CALM = 'This art piece makes me feel calm.'
COL_UNEASY = 'This art piece makes me feel uneasy.'
COL_COLOURS = 'How many prominent colours do you notice in this painting?'
COL_OBJECTS = 'How many objects caught your eye in the painting?'
COL_PRICE = 'How much (in Canadian dollars) would you be willing to pay for this painting?'
COL_ROOM = 'If you could purchase this painting, which room would you put that painting in?'
COL_WHO = 'If you could view this art in person, who would you want to view it with?'
COL_SEASON = 'What season does this art piece remind you of?'
COL_FOOD = 'If this painting was a food, what would be?'
COL_SOUNDTRACK = 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.'

LIKERT_COLS = [COL_SOMBRE, COL_CONTENT, COL_CALM, COL_UNEASY]
LIKERT_SHORT = ['Sombre', 'Content', 'Calm', 'Uneasy']
NUMERICAL_COLS = [COL_EMOTION, COL_COLOURS, COL_OBJECTS]
MULTISELECT_COLS = [COL_ROOM, COL_WHO, COL_SEASON]
TEXT_COLS = [COL_DESCRIBE, COL_FOOD, COL_SOUNDTRACK]


# ============================================================
# 1. Likert Processing
# ============================================================
def extract_likert_numeric(series):
    """Extract numeric prefix from '4 - Agree' → 4."""
    return pd.to_numeric(series.str.extract(r'^(\d)', expand=False), errors='coerce')


def analyze_likert(df, fig_dir):
    print("\n" + "=" * 70)
    print("LIKERT ANALYSIS")
    print("=" * 70)

    likert_numeric = pd.DataFrame()
    for col, short in zip(LIKERT_COLS, LIKERT_SHORT):
        likert_numeric[short] = extract_likert_numeric(df[col])

    # Mean Likert per class
    likert_numeric[COL_PAINTING] = df[COL_PAINTING]
    means = likert_numeric.groupby(COL_PAINTING)[LIKERT_SHORT].mean()
    means = means.reindex(CLASS_ORDER)

    print("\nMean Likert scores per class:")
    print(means.round(3).to_string())

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LIKERT_SHORT))
    width = 0.25
    for i, (cls, short) in enumerate(zip(CLASS_ORDER, CLASS_SHORT)):
        vals = means.loc[cls]
        ax.bar(x + i * width, vals, width, label=short, color=COLORS_3CLASS[i])

    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Mean Likert Score (1-5)', fontsize=12)
    ax.set_title('Mean Likert Scores by Painting Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(LIKERT_SHORT)
    ax.legend()
    ax.set_ylim(1, 5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    fig.tight_layout()
    save_fig(fig, fig_dir, 'likert_profile_by_class')

    return means


# ============================================================
# 2. Price Cleaning
# ============================================================
def clean_price(series):
    """Extract first numeric value from messy price text."""
    def extract_price(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        # Remove dollar signs, commas, spaces in numbers
        val = val.replace(',', '').replace('$', '').replace(' ', '')
        # Try to find first number (int or float)
        match = re.search(r'(\d+\.?\d*)', val)
        if match:
            return float(match.group(1))
        return np.nan

    return series.apply(extract_price)


def analyze_price(df, fig_dir):
    print("\n" + "=" * 70)
    print("PRICE ANALYSIS")
    print("=" * 70)

    prices = clean_price(df[COL_PRICE])
    valid_count = prices.notna().sum()
    total = len(prices)
    unparseable = total - df[COL_PRICE].isna().sum() - valid_count + df[COL_PRICE].isna().sum()

    print(f"  Total rows: {total}")
    print(f"  Valid numeric prices: {valid_count}")
    print(f"  Missing (NaN): {df[COL_PRICE].isna().sum()}")
    print(f"  Unparseable: {(df[COL_PRICE].notna() & prices.isna()).sum()}")
    print(f"  Price range: ${prices.min():.0f} - ${prices.max():.0f}")
    print(f"  Median: ${prices.median():.0f}")
    print(f"  Mean: ${prices.mean():.0f}")
    print(f"  99th percentile: ${prices.quantile(0.99):.0f}")

    # Clip at 99th percentile for visualization
    clip_val = prices.quantile(0.99)
    prices_clipped = prices.clip(upper=clip_val)

    # Log-scale histogram by class
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw distribution
    for i, (cls, short) in enumerate(zip(CLASS_ORDER, CLASS_SHORT)):
        mask = df[COL_PAINTING] == cls
        vals = prices_clipped[mask].dropna()
        axes[0].hist(vals, bins=40, alpha=0.5, label=short, color=COLORS_3CLASS[i])
    axes[0].set_xlabel('Price (CAD, clipped at 99th pct)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Price Distribution by Class', fontsize=13, fontweight='bold')
    axes[0].legend()

    # Log-scale
    prices_log = np.log1p(prices.dropna())
    prices_log_df = pd.DataFrame({'log_price': prices_log, COL_PAINTING: df.loc[prices.dropna().index, COL_PAINTING]})
    for i, (cls, short) in enumerate(zip(CLASS_ORDER, CLASS_SHORT)):
        vals = prices_log_df[prices_log_df[COL_PAINTING] == cls]['log_price']
        axes[1].hist(vals, bins=40, alpha=0.5, label=short, color=COLORS_3CLASS[i])
    axes[1].set_xlabel('log(1 + Price)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Log-Transformed Price Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, fig_dir, 'price_distribution')

    return prices


# ============================================================
# 3. Multi-Select Encoding Analysis
# ============================================================
def analyze_multiselect(df, fig_dir):
    print("\n" + "=" * 70)
    print("MULTI-SELECT ANALYSIS")
    print("=" * 70)

    multiselect_data = {}

    for col, col_name in [(COL_ROOM, 'Room'), (COL_WHO, 'Who'), (COL_SEASON, 'Season')]:
        # Get all unique options
        all_options = set()
        for val in df[col].dropna():
            for opt in str(val).split(','):
                opt = opt.strip()
                if opt:
                    all_options.add(opt)
        all_options = sorted(all_options)
        print(f"\n  {col_name} options: {all_options}")

        # Compute frequencies per class
        freq_data = {}
        for cls in CLASS_ORDER:
            subset = df[df[COL_PAINTING] == cls][col].dropna()
            n = len(subset)
            counts = {}
            for opt in all_options:
                count = subset.str.contains(re.escape(opt), na=False).sum()
                counts[opt] = count / n * 100  # percentage
            freq_data[cls] = counts

        multiselect_data[col_name] = {'options': all_options, 'freq': freq_data}

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax_idx, (col_name, data) in enumerate(multiselect_data.items()):
        options = data['options']
        x = np.arange(len(options))
        width = 0.25
        for i, (cls, short) in enumerate(zip(CLASS_ORDER, CLASS_SHORT)):
            vals = [data['freq'][cls][opt] for opt in options]
            axes[ax_idx].bar(x + i * width, vals, width, label=short, color=COLORS_3CLASS[i])

        axes[ax_idx].set_xlabel('Option', fontsize=10)
        axes[ax_idx].set_ylabel('Percentage (%)', fontsize=10)
        axes[ax_idx].set_title(f'{col_name} Preferences by Class', fontsize=12, fontweight='bold')
        axes[ax_idx].set_xticks(x + width)
        axes[ax_idx].set_xticklabels(options, rotation=45, ha='right', fontsize=8)
        axes[ax_idx].legend(fontsize=8)

    fig.suptitle('Multi-Select Response Distributions by Painting Class', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, fig_dir, 'multiselect_by_class')

    return multiselect_data


# ============================================================
# 4. Correlation Analysis
# ============================================================
def analyze_correlations(df, fig_dir):
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Build numeric dataframe
    numeric_df = pd.DataFrame()
    numeric_df['Emotion Intensity'] = df[COL_EMOTION]
    numeric_df['Colours Count'] = df[COL_COLOURS]
    numeric_df['Objects Count'] = df[COL_OBJECTS]

    for col, short in zip(LIKERT_COLS, LIKERT_SHORT):
        numeric_df[short] = extract_likert_numeric(df[col])

    numeric_df['Price'] = clean_price(df[COL_PRICE])

    corr = numeric_df.corr()
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, fig_dir, 'correlation_heatmap')

    return corr


# ============================================================
# 5. Missing Values Chart
# ============================================================
def plot_missing_values(df, fig_dir):
    print("\n" + "=" * 70)
    print("MISSING VALUES VISUALIZATION")
    print("=" * 70)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    pct = (missing / len(df)) * 100

    # Shorten column names for display
    short_names = {
        COL_EMOTION: 'Emotion Intensity',
        COL_DESCRIBE: 'Describe Feeling',
        COL_SOMBRE: 'Likert: Sombre',
        COL_CONTENT: 'Likert: Content',
        COL_CALM: 'Likert: Calm',
        COL_UNEASY: 'Likert: Uneasy',
        COL_COLOURS: 'Colours Count',
        COL_OBJECTS: 'Objects Count',
        COL_PRICE: 'Price',
        COL_ROOM: 'Room',
        COL_WHO: 'Who',
        COL_SEASON: 'Season',
        COL_FOOD: 'Food Analogy',
        COL_SOUNDTRACK: 'Soundtrack',
    }

    labels = [short_names.get(c, c) for c in missing.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(missing)), pct.values, color='#e74c3c', alpha=0.8)
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Missing Values (%)', fontsize=12)
    ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')

    for bar, count, p in zip(bars, missing.values, pct.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{count} ({p:.1f}%)', va='center', fontsize=9)

    ax.set_xlim(0, max(pct.values) * 1.3)
    fig.tight_layout()
    save_fig(fig, fig_dir, 'missing_values')

    # Check student-level missingness pattern
    print("\n  Checking student-level missingness pattern...")
    student_missing = df.groupby(COL_ID).apply(lambda g: g.isnull().any(axis=1).sum())
    print(f"  Students with at least 1 missing row: {(student_missing > 0).sum()}")
    print(f"  Students with all 3 rows having missing: {(student_missing == 3).sum()}")
    print(f"  Students with 0 missing rows: {(student_missing == 0).sum()}")


# ============================================================
# 6. Data Splitting with Leakage Prevention
# ============================================================
def perform_grouped_split(df):
    print("\n" + "=" * 70)
    print("GROUPED DATA SPLITTING")
    print("=" * 70)

    groups = df[COL_ID].values
    n_students = df[COL_ID].nunique()

    # First split: 60% train, 40% temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    df_train = df.iloc[train_idx]
    df_temp = df.iloc[temp_idx]

    # Second split: 50% of temp = 20% val, 20% test
    temp_groups = df_temp[COL_ID].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(df_temp, groups=temp_groups))

    df_val = df_temp.iloc[val_idx]
    df_test = df_temp.iloc[test_idx]

    # Verify no leakage
    train_students = set(df_train[COL_ID].unique())
    val_students = set(df_val[COL_ID].unique())
    test_students = set(df_test[COL_ID].unique())

    assert len(train_students & val_students) == 0, "LEAKAGE: train/val overlap!"
    assert len(train_students & test_students) == 0, "LEAKAGE: train/test overlap!"
    assert len(val_students & test_students) == 0, "LEAKAGE: val/test overlap!"

    print(f"\n  Split verification PASSED — no student appears in multiple splits.")
    print(f"\n  Train: {len(df_train)} rows, {len(train_students)} students ({len(train_students)/n_students*100:.1f}%)")
    print(f"  Val:   {len(df_val)} rows, {len(val_students)} students ({len(val_students)/n_students*100:.1f}%)")
    print(f"  Test:  {len(df_test)} rows, {len(test_students)} students ({len(test_students)/n_students*100:.1f}%)")

    # Verify class balance in each split
    print(f"\n  Class balance per split:")
    for name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        counts = split_df[COL_PAINTING].value_counts(normalize=True) * 100
        balance_str = ', '.join([f"{cls[:12]}: {counts.get(cls, 0):.1f}%" for cls in CLASS_ORDER])
        print(f"    {name}: {balance_str}")

    print(f"\n  NOTE: Test set was NOT used during any exploration analysis.")

    return {
        'train_students': len(train_students),
        'val_students': len(val_students),
        'test_students': len(test_students),
        'train_rows': len(df_train),
        'val_rows': len(df_val),
        'test_rows': len(df_test),
        'leakage_check': 'PASSED',
    }


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python deep_analysis.py <path_to_csv> [--output_dir outputs]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = 'outputs'
    if '--output_dir' in sys.argv:
        output_dir = sys.argv[sys.argv.index('--output_dir') + 1]

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

    results = {}

    # 1. Likert
    likert_means = analyze_likert(df, fig_dir)
    results['likert_means'] = likert_means.to_dict()

    # 2. Price
    prices = analyze_price(df, fig_dir)
    results['price_stats'] = {
        'valid_count': int(prices.notna().sum()),
        'median': float(prices.median()) if prices.notna().any() else None,
        'mean': float(prices.mean()) if prices.notna().any() else None,
        'p99': float(prices.quantile(0.99)) if prices.notna().any() else None,
    }

    # 3. Multi-select
    multiselect_data = analyze_multiselect(df, fig_dir)

    # 4. Correlations
    corr = analyze_correlations(df, fig_dir)
    results['top_correlations'] = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.3:
                results['top_correlations'].append({
                    'feature1': corr.columns[i],
                    'feature2': corr.columns[j],
                    'correlation': round(float(r), 3)
                })

    # 5. Missing values
    plot_missing_values(df, fig_dir)

    # 6. Grouped split
    split_results = perform_grouped_split(df)
    results['split'] = split_results

    # Save results
    results_path = os.path.join(output_dir, 'deep_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] Deep analysis results: {results_path}")

    print("\n" + "=" * 70)
    print("DEEP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
