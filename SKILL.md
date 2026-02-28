---
name: csc311-data-exploration
description: >
  Perform thorough, structured data exploration for the CSC311 ML classification project.
  Use this skill whenever the user asks to explore a dataset for a machine learning project,
  perform EDA (exploratory data analysis), analyze features, check for data issues, plan
  preprocessing, or generate data exploration reports. Also trigger when the user mentions
  CSC311, project proposal, data exploration section, or asks about class balance, missing
  values, outliers, text features, or data splitting strategies. This skill produces
  publication-ready analysis with figures, following the CSC311 report template requirements.
---

# CSC311 Data Exploration Skill

You are an expert data scientist performing exploratory data analysis for a university
ML classification project. The dataset contains student responses that must be classified
into one of three categories. Your analysis must be rigorous enough for a graded academic
report.

## Overview

This skill guides a complete data exploration workflow that produces:
1. A structured analysis covering all 6 required sections
2. Publication-quality figures saved as PNG files
3. A markdown summary report suitable for inclusion in a LaTeX document
4. Concrete recommendations connecting findings to model selection

## Workflow

Execute these phases **in order**. Complete each phase fully before moving to the next.

### Phase 1: Initial Data Audit

Load the CSV and produce a comprehensive first look:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("<path_to_csv>")

# Shape and structure
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe(include='all')}")
```

Classify every column into one of:
- **Numerical**: continuous or discrete numeric values
- **Categorical**: finite set of labels/categories
- **Text**: free-form text responses (likely the main features)
- **Identifier**: student IDs, row indices (not features)

Record your classification in a summary table.

### Phase 2: Dataset Summary (Report Section 1)

For each feature type, analyze distributions:

**Numerical features:**
- Compute mean, median, std, min, max, skewness
- Generate histograms with KDE overlays
- Note any bimodal or heavily skewed distributions

**Categorical features:**
- Compute value counts and proportions
- Generate bar charts

**Text features:**
- Compute document lengths (word count, character count)
- Compute vocabulary size and most frequent terms
- Generate word frequency bar plots
- Optionally generate word clouds (if wordcloud is installed)

**Class balance:**
- Compute class distribution (counts and percentages)
- Generate a class distribution bar chart
- Assess whether classes are balanced, slightly imbalanced, or heavily imbalanced
- If imbalanced, note implications for model training and evaluation metrics

Save all figures to an `outputs/figures/` directory with descriptive filenames.

### Phase 3: Data Issues (Report Section 2)

Systematically check for and document:

**Missing values:**
```python
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({'count': missing, 'pct': missing_pct})
print(missing_report[missing_report['count'] > 0])
```
For each column with missing values, recommend a handling strategy:
- Drop rows (if very few)
- Impute with mean/median/mode (if moderate)
- Create a "missing" indicator feature (if missingness is informative)

**Outliers:**
- For numerical columns, use IQR method and z-scores
- Visualize with box plots
- Recommend: keep, clip, or remove — with justification

**Inconsistencies:**
- For text: check for encoding issues, extra whitespace, HTML artifacts
- For categorical: check for near-duplicate labels (case differences, typos)
- For all: check for duplicate rows

**Noise:**
- For text: check for very short or empty responses that may be low-effort
- Flag any suspicious patterns

For every issue found, state:
1. What the issue is
2. How many rows/values are affected
3. Your recommended handling approach and why

### Phase 4: Preprocessing Plan (Report Section 3)

Based on findings from Phases 2-3, outline a concrete preprocessing pipeline:

**Text representation** (most likely the core preprocessing):
- Tokenization approach (word-level, subword)
- Lowercasing, punctuation removal, stop word removal
- Representation: TF-IDF, Bag-of-Words, or embeddings
- Justify the choice based on vocabulary size and document lengths observed

**Numerical features:**
- StandardScaler vs MinMaxScaler — choose based on distribution shape
- Handle outliers before or after scaling

**Categorical features:**
- One-hot encoding vs label encoding — choose based on cardinality

**Feature engineering ideas** (optional but valuable):
- Text length as a feature
- Sentiment scores
- N-gram features
- PCA for dimensionality reduction on high-dimensional text features

### Phase 5: Data Splitting Strategy (Report Section 4)

This is critical for correctness. Implement and explain:

```python
from sklearn.model_selection import GroupShuffleSplit

# Identify the student ID column (the grouping key)
student_id_col = "<student_id_column>"  # Determine this from the data

# All 3 data points from the same student MUST stay in the same split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss.split(df, groups=df[student_id_col]))

# Further split train_val into train and validation
df_train_val = df.iloc[train_val_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2
train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val[student_id_col]))
```

You MUST explicitly address:
- **Why grouped splitting**: each student contributed 3 related data points (one per class).
  If split across train/val/test, the model could learn student-specific patterns instead
  of generalizable class patterns. This is data leakage.
- **The split ratios**: e.g., 60/20/20 or 70/15/15 — justify the choice
- **That the test set is reserved and untouched** during all exploration and model development
- **Verify no leakage**: confirm no student ID appears in multiple splits

```python
# Verification
train_students = set(df.iloc[train_idx][student_id_col])
val_students = set(df.iloc[val_idx][student_id_col])
test_students = set(df.iloc[test_idx][student_id_col])
assert len(train_students & val_students) == 0, "Leakage between train and val!"
assert len(train_students & test_students) == 0, "Leakage between train and test!"
assert len(val_students & test_students) == 0, "Leakage between val and test!"
print("No leakage detected.")
```

### Phase 6: Connect to Model Choices (Report Section 5)

Based on everything discovered, write a section connecting findings to model selection.
Use this decision framework:

| Finding | Suggested Model Family | Reasoning |
|---------|----------------------|-----------|
| High-dimensional sparse text (TF-IDF) | Linear models (Logistic Regression, Linear SVM) | Effective in high-dim sparse spaces |
| Nonlinear patterns in numerical features | Tree-based (Random Forest, Gradient Boosting) | Capture nonlinear decision boundaries |
| Small dataset size | Simpler models, strong regularization | Avoid overfitting |
| Class imbalance | Models with class_weight support | Handle imbalance natively |
| Mixed feature types | Tree-based or ensemble methods | Handle heterogeneous features naturally |
| Complex text semantics | Neural approaches (if dataset is large enough) | Capture semantic meaning |

Be specific — don't just list models generically. Connect each recommendation to a
concrete observation from your analysis.

### Phase 7: Generate the Report

After all analysis is complete, compile everything into a structured markdown report:

```
# Data Exploration

## 1. Dataset Summary
[Feature types table, distribution descriptions, class balance analysis]
[Reference figures by filename]

## 2. Data Issues
[Missing values, outliers, inconsistencies — with handling strategies]

## 3. Preprocessing Plan
[Transformations with justifications]

## 4. Data Splitting Strategy
[Grouped splitting explanation, leakage prevention, verification]

## 5. Key Insights and Model Choice Connections
[Findings → model family recommendations]
```

Save this report as `outputs/data_exploration_report.md`.

## Figure Standards

All figures must:
- Have descriptive titles
- Have labeled axes with units where applicable
- Use a consistent color palette (e.g., `tab10` or a custom 3-color scheme for the 3 classes)
- Be saved at 300 DPI as PNG
- Be sized appropriately (typically 8x6 or 10x6 inches)
- Include a brief caption/explanation in the report

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red for 3 classes
FIG_DIR = 'outputs/figures/'
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
```

## Common Pitfalls to Avoid

- Do NOT use the test set for any exploration, visualization, or statistics. All analysis
  uses the full dataset BEFORE splitting, or the training set AFTER splitting. State this
  explicitly.
- Do NOT just paste plots without interpretation. Every figure needs 1-3 sentences explaining
  what it shows and why it matters.
- Do NOT skip the leakage verification step. This is specifically called out in the rubric.
- Do NOT present surface-level analysis. "There are no missing values" is fine if true, but
  also check for implicit missing values (empty strings, "N/A", "none", etc.).
- Do NOT forget to connect findings to model choices. This is what separates a good report
  from a great one.

## Additional References

For detailed preprocessing guidance for specific feature types, see:
- `references/text_preprocessing.md` — detailed text feature analysis patterns
- `references/reporting_checklist.md` — final checklist before submission
