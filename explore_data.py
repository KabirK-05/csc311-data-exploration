"""
CSC311 Data Exploration Script
==============================
Automated exploratory data analysis for the CSC311 ML classification project.
Claude Code should execute this script, then interpret and expand on the results.

Usage:
    python explore_data.py <path_to_csv> [--output_dir outputs]

This script produces:
    - outputs/figures/*.png  — all visualization figures
    - outputs/data_audit.json — structured audit results
    - Console output summarizing all findings
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import re
from collections import Counter

# ============================================================
# Configuration
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
COLORS_3CLASS = ['#2ecc71', '#3498db', '#e74c3c']
sns.set_palette(COLORS_3CLASS)
DPI = 300
FIGSIZE = (10, 6)

def setup_output_dir(output_dir):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

def save_fig(fig, fig_dir, name):
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path

# ============================================================
# Phase 1: Data Audit
# ============================================================
def run_data_audit(df):
    print("=" * 70)
    print("PHASE 1: DATA AUDIT")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns and types:")
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
        print(f"  {col:30s} | dtype: {str(dtype):10s} | unique: {nunique:6d} | sample: {str(sample)[:50]}")

    # Auto-classify columns
    classifications = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].nunique() < 10:
                classifications[col] = 'categorical (low-cardinality numeric)'
            else:
                classifications[col] = 'numerical'
        elif df[col].dtype == 'object':
            avg_len = df[col].dropna().str.len().mean() if len(df[col].dropna()) > 0 else 0
            if avg_len > 50:
                classifications[col] = 'text'
            elif df[col].nunique() < 20:
                classifications[col] = 'categorical'
            else:
                classifications[col] = 'text (or high-cardinality categorical)'
        else:
            classifications[col] = 'other'

    print(f"\nAuto-classified feature types:")
    for col, ftype in classifications.items():
        print(f"  {col:30s} → {ftype}")

    return classifications

# ============================================================
# Phase 2: Dataset Summary
# ============================================================
def analyze_class_balance(df, fig_dir):
    print("\n" + "=" * 70)
    print("PHASE 2: CLASS BALANCE ANALYSIS")
    print("=" * 70)

    # Try to find the label/target column
    label_candidates = [c for c in df.columns if c.lower() in
                        ['label', 'class', 'target', 'category', 'y']]
    if not label_candidates:
        # Look for columns with exactly 3 unique values
        label_candidates = [c for c in df.columns if df[c].nunique() == 3]

    if label_candidates:
        label_col = label_candidates[0]
        print(f"\nDetected label column: '{label_col}'")
    else:
        print("\n⚠ Could not auto-detect label column. Listing candidates:")
        for col in df.columns:
            print(f"  {col}: {df[col].nunique()} unique values")
        return None

    counts = df[label_col].value_counts()
    pcts = df[label_col].value_counts(normalize=True) * 100

    print(f"\nClass distribution:")
    for label in counts.index:
        print(f"  {label}: {counts[label]} ({pcts[label]:.1f}%)")

    # Balance assessment
    max_pct, min_pct = pcts.max(), pcts.min()
    imbalance_ratio = max_pct / min_pct
    if imbalance_ratio < 1.2:
        balance_status = "BALANCED"
    elif imbalance_ratio < 2.0:
        balance_status = "SLIGHTLY IMBALANCED"
    else:
        balance_status = "IMBALANCED"
    print(f"\nBalance assessment: {balance_status} (ratio: {imbalance_ratio:.2f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    bars = axes[0].bar(range(len(counts)), counts.values, color=COLORS_3CLASS[:len(counts)])
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=0)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution (Counts)')
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     str(count), ha='center', va='bottom', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, colors=COLORS_3CLASS[:len(counts)],
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Distribution (Proportions)')

    fig.suptitle('Class Balance Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, fig_dir, 'class_distribution')

    return label_col

def analyze_numerical_features(df, classifications, fig_dir):
    num_cols = [c for c, t in classifications.items() if t == 'numerical']
    if not num_cols:
        print("\nNo numerical features detected.")
        return

    print(f"\n--- Numerical Features ({len(num_cols)}) ---")
    for col in num_cols:
        stats = df[col].describe()
        skew = df[col].skew()
        print(f"\n  {col}:")
        print(f"    Mean={stats['mean']:.3f}, Median={stats['50%']:.3f}, Std={stats['std']:.3f}")
        print(f"    Min={stats['min']:.3f}, Max={stats['max']:.3f}, Skewness={skew:.3f}")

    # Histograms
    n = len(num_cols)
    if n > 0:
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, col in enumerate(num_cols):
            axes[i].hist(df[col].dropna(), bins=30, color=COLORS_3CLASS[0], edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Numerical Feature Distributions', fontsize=14, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, fig_dir, 'numerical_distributions')

def analyze_text_features(df, classifications, label_col, fig_dir):
    text_cols = [c for c, t in classifications.items() if 'text' in t.lower()]
    if not text_cols:
        print("\nNo text features detected.")
        return

    print(f"\n--- Text Features ({len(text_cols)}) ---")
    for col in text_cols:
        series = df[col].fillna('')
        word_counts = series.str.split().str.len()
        char_counts = series.str.len()
        all_words = ' '.join(series).lower().split()
        vocab = set(all_words)

        print(f"\n  {col}:")
        print(f"    Documents: {len(series)}")
        print(f"    Empty/null: {(series == '').sum() + df[col].isna().sum()}")
        print(f"    Avg words: {word_counts.mean():.1f} (median: {word_counts.median():.1f})")
        print(f"    Word range: [{word_counts.min()}, {word_counts.max()}]")
        print(f"    Avg chars: {char_counts.mean():.1f}")
        print(f"    Vocabulary size: {len(vocab)}")
        print(f"    Top 10 words: {Counter(all_words).most_common(10)}")

        # Document length histogram by class
        if label_col and label_col in df.columns:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            for idx, label in enumerate(sorted(df[label_col].unique())):
                subset = df[df[label_col] == label][col].fillna('').str.split().str.len()
                ax.hist(subset, bins=30, alpha=0.5, label=str(label),
                        color=COLORS_3CLASS[idx % len(COLORS_3CLASS)])
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Document Length Distribution by Class — {col}')
            ax.legend()
            save_fig(fig, fig_dir, f'text_length_by_class_{col}')

        # Top words per class
        if label_col and label_col in df.columns:
            classes = sorted(df[label_col].unique())
            n_classes = len(classes)
            fig, axes = plt.subplots(1, n_classes, figsize=(7*n_classes, 6))
            if n_classes == 1:
                axes = [axes]

            for idx, label in enumerate(classes):
                subset_text = df[df[label_col] == label][col].fillna('')
                words = ' '.join(subset_text).lower().split()
                # Remove very short words and common stop words
                stop = {'the', 'a', 'an', 'is', 'it', 'to', 'of', 'and', 'in', 'for',
                        'on', 'that', 'this', 'with', 'as', 'are', 'was', 'be', 'or',
                        'at', 'by', 'from', 'not', 'but', 'they', 'have', 'has', 'had',
                        'i', 'you', 'we', 'my', 'your', 'can', 'will', 'do', 'does'}
                words = [w for w in words if len(w) > 2 and w not in stop]
                top = Counter(words).most_common(15)

                if top:
                    words_list, counts_list = zip(*top)
                    axes[idx].barh(range(len(words_list)), counts_list,
                                   color=COLORS_3CLASS[idx % len(COLORS_3CLASS)])
                    axes[idx].set_yticks(range(len(words_list)))
                    axes[idx].set_yticklabels(words_list)
                    axes[idx].invert_yaxis()
                    axes[idx].set_xlabel('Frequency')
                    axes[idx].set_title(f'Top Words — Class: {label}')

            fig.suptitle(f'Most Frequent Words per Class — {col}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            save_fig(fig, fig_dir, f'top_words_per_class_{col}')

# ============================================================
# Phase 3: Data Issues
# ============================================================
def check_data_issues(df, classifications, fig_dir):
    print("\n" + "=" * 70)
    print("PHASE 3: DATA ISSUES")
    print("=" * 70)

    issues = []

    # --- Missing values ---
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    has_missing = missing[missing > 0]
    if len(has_missing) > 0:
        for col in has_missing.index:
            msg = f"  {col}: {has_missing[col]} missing ({missing_pct[col]:.1f}%)"
            print(msg)
            issues.append({'type': 'missing', 'column': col,
                          'count': int(has_missing[col]), 'pct': float(missing_pct[col])})
    else:
        print("  No explicit missing values (NaN/None) found.")

    # Check for implicit missing values
    print("\n--- Implicit Missing Values ---")
    for col in df.select_dtypes(include='object').columns:
        empty_strings = (df[col].str.strip() == '').sum()
        na_strings = df[col].str.lower().isin(['n/a', 'na', 'none', 'null', '-', 'nan', '']).sum()
        if empty_strings > 0 or na_strings > 0:
            msg = f"  {col}: {empty_strings} empty strings, {na_strings} NA-like strings"
            print(msg)
            issues.append({'type': 'implicit_missing', 'column': col,
                          'empty_strings': int(empty_strings), 'na_like': int(na_strings)})

    # --- Duplicates ---
    print("\n--- Duplicates ---")
    n_dupes = df.duplicated().sum()
    print(f"  Exact duplicate rows: {n_dupes}")
    if n_dupes > 0:
        issues.append({'type': 'duplicates', 'count': int(n_dupes)})

    # --- Outliers (numerical) ---
    print("\n--- Outliers (IQR Method) ---")
    num_cols = [c for c, t in classifications.items() if t == 'numerical']
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} outliers (below {lower:.2f} or above {upper:.2f})")
            issues.append({'type': 'outlier', 'column': col, 'count': int(n_outliers)})

    if num_cols:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(6*len(num_cols), 5))
        if len(num_cols) == 1:
            axes = [axes]
        for i, col in enumerate(num_cols):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'{col}')
        fig.suptitle('Box Plots for Outlier Detection', fontsize=14, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, fig_dir, 'outlier_boxplots')

    # --- Text inconsistencies ---
    print("\n--- Text Inconsistencies ---")
    text_cols = [c for c, t in classifications.items() if 'text' in t.lower()]
    for col in text_cols:
        series = df[col].dropna()
        # Check for HTML tags
        html_count = series.str.contains(r'<[^>]+>', regex=True, na=False).sum()
        if html_count > 0:
            print(f"  {col}: {html_count} entries contain HTML tags")
            issues.append({'type': 'html_artifacts', 'column': col, 'count': int(html_count)})
        # Check for URLs
        url_count = series.str.contains(r'http[s]?://\S+', regex=True, na=False).sum()
        if url_count > 0:
            print(f"  {col}: {url_count} entries contain URLs")
            issues.append({'type': 'urls', 'column': col, 'count': int(url_count)})
        # Check for very short responses
        short = (series.str.split().str.len() < 3).sum()
        if short > 0:
            print(f"  {col}: {short} entries with fewer than 3 words (potentially low-effort)")
            issues.append({'type': 'short_response', 'column': col, 'count': int(short)})

    print(f"\nTotal issues found: {len(issues)}")
    return issues

# ============================================================
# Phase 4: Student Grouping Analysis
# ============================================================
def analyze_student_groups(df):
    print("\n" + "=" * 70)
    print("PHASE 4: STUDENT GROUPING ANALYSIS (for data splitting)")
    print("=" * 70)

    # Try to find student ID column
    id_candidates = [c for c in df.columns if any(kw in c.lower()
                     for kw in ['student', 'user', 'id', 'respondent', 'contributor'])]

    if id_candidates:
        id_col = id_candidates[0]
    else:
        # Heuristic: column with ~1/3 of total rows unique values (3 responses per student)
        for col in df.columns:
            ratio = df[col].nunique() / len(df)
            if 0.25 < ratio < 0.4:
                id_candidates.append(col)
        id_col = id_candidates[0] if id_candidates else None

    if id_col:
        print(f"\nDetected student ID column: '{id_col}'")
        n_students = df[id_col].nunique()
        responses_per_student = df[id_col].value_counts()
        print(f"  Unique students: {n_students}")
        print(f"  Responses per student: min={responses_per_student.min()}, "
              f"max={responses_per_student.max()}, "
              f"median={responses_per_student.median():.0f}, "
              f"mode={responses_per_student.mode().values[0]}")
        if responses_per_student.mode().values[0] == 3:
            print("  ✓ Confirms: most students have exactly 3 responses (one per class)")
        students_not_3 = (responses_per_student != 3).sum()
        if students_not_3 > 0:
            print(f"  ⚠ {students_not_3} students do NOT have exactly 3 responses")
    else:
        print("\n⚠ Could not auto-detect student ID column.")
        print("  You MUST identify this column for proper grouped splitting.")
        print("  Look for a column where each value appears ~3 times.")

    return id_col

# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_data.py <path_to_csv> [--output_dir outputs]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = 'outputs'
    if '--output_dir' in sys.argv:
        output_dir = sys.argv[sys.argv.index('--output_dir') + 1]

    fig_dir = setup_output_dir(output_dir)

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Phase 1
    classifications = run_data_audit(df)

    # Phase 2
    label_col = analyze_class_balance(df, fig_dir)
    analyze_numerical_features(df, classifications, fig_dir)
    analyze_text_features(df, classifications, label_col, fig_dir)

    # Phase 3
    issues = check_data_issues(df, classifications, fig_dir)

    # Phase 4
    id_col = analyze_student_groups(df)

    # Save structured results
    audit_results = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'classifications': classifications,
        'label_column': label_col,
        'student_id_column': id_col,
        'issues': issues,
        'class_distribution': df[label_col].value_counts().to_dict() if label_col else None,
    }

    audit_path = os.path.join(output_dir, 'data_audit.json')
    with open(audit_path, 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)
    print(f"\n[SAVED] Structured audit results: {audit_path}")

    print("\n" + "=" * 70)
    print("DATA EXPLORATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"Figures saved to: {fig_dir}/")
    print("\nNext steps for Claude Code:")
    print("  1. Review the figures and audit results")
    print("  2. Write interpretations for each finding")
    print("  3. Draft the preprocessing plan based on identified issues")
    print("  4. Implement grouped data splitting")
    print("  5. Connect findings to model family recommendations")
    print("  6. Compile the final data exploration report")

if __name__ == '__main__':
    main()
