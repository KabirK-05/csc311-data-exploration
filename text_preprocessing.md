# Text Preprocessing Reference

This reference covers detailed text preprocessing strategies for the CSC311 project.
Read this file when the dataset contains text/free-response features.

## Text Analysis Pipeline

### Step 1: Raw Text Statistics

Before any preprocessing, compute baseline statistics:

```python
def text_stats(series):
    """Compute comprehensive text statistics for a pandas Series of strings."""
    stats = {
        'total_documents': len(series),
        'empty_or_null': series.isna().sum() + (series.str.strip() == '').sum(),
        'avg_word_count': series.str.split().str.len().mean(),
        'median_word_count': series.str.split().str.len().median(),
        'min_word_count': series.str.split().str.len().min(),
        'max_word_count': series.str.split().str.len().max(),
        'avg_char_count': series.str.len().mean(),
        'unique_words': len(set(' '.join(series.dropna()).lower().split())),
    }
    return stats
```

### Step 2: Text Cleaning

Apply these in order:
1. **Lowercase** — reduces vocabulary size without losing meaning for classification
2. **Strip whitespace** — remove leading/trailing and collapse multiple spaces
3. **Remove URLs** — `re.sub(r'http\S+|www\.\S+', '', text)`
4. **Remove HTML tags** — `re.sub(r'<[^>]+>', '', text)`
5. **Handle punctuation** — remove or keep depending on task (for classification, usually remove)
6. **Handle numbers** — replace with `<NUM>` token or remove

### Step 3: Tokenization

For a course project with sklearn, word-level tokenization is standard:
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF is generally preferred for classification
tfidf = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary size
    min_df=2,               # Ignore very rare terms
    max_df=0.95,            # Ignore terms in >95% of documents
    ngram_range=(1, 2),     # Unigrams and bigrams
    stop_words='english',
    sublinear_tf=True       # Apply log normalization
)
```

### Step 4: Choosing Text Representation

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| Bag-of-Words | Baseline, simple models | Simple, interpretable | Ignores word order |
| TF-IDF | Most classification tasks | Weights important words | Still ignores order |
| N-grams (1,2) | When phrases matter | Captures some context | Much higher dimensionality |
| Truncated SVD after TF-IDF | High dimensionality | Dense, lower-dim | Loses interpretability |

For this project, TF-IDF with (1,2)-grams is a strong default choice. Justify by noting:
- Document lengths (if short, unigrams may suffice)
- Vocabulary size (if very large, use max_features or SVD)
- Whether specific phrases are discriminative between classes

### Step 5: Visualizations for Text

**Word frequency plots** (top 20 words per class):
```python
from collections import Counter

for label in df['label'].unique():
    subset = df[df['label'] == label]['text_column']
    words = ' '.join(subset).lower().split()
    common = Counter(words).most_common(20)
    # Plot horizontal bar chart
```

**Document length distribution by class:**
```python
fig, ax = plt.subplots(figsize=(10, 6))
for label in df['label'].unique():
    subset = df[df['label'] == label]['text_column']
    ax.hist(subset.str.split().str.len(), alpha=0.5, label=label, bins=30)
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.set_title('Document Length Distribution by Class')
ax.legend()
```

**TF-IDF feature importance** (top features per class):
After fitting a simple logistic regression on TF-IDF features, extract and plot the
highest-weighted features per class. This provides strong evidence for model choice connections.

### Step 6: Text-Specific Issues to Check

- **Empty responses**: flag and decide whether to impute or drop
- **Very short responses** (<5 words): may be noise or low-effort
- **Copy-pasted content**: check for exact duplicate text across different students
- **Language mixing**: check if responses are consistently in one language
- **Special characters**: emojis, unicode, LaTeX notation
- **Encoding issues**: garbled characters from encoding mismatches
