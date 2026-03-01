# Section 2: Data Exploration

Understanding the dataset's structure, distributions, and quality issues is essential for selecting appropriate models and preprocessing strategies. This section presents our exploratory data analysis findings and the decisions they inform.

## 2.1 Dataset Summary

The dataset contains **1,686 rows** and **16 columns** representing survey responses from **562 students**, each of whom evaluated three paintings. The three paintings (class labels) are *The Persistence of Memory*, *The Starry Night*, and *The Water Lily Pond*, with exactly **562 observations per class (33.3% each)**---a perfectly balanced dataset (Figure 1). This balance means accuracy is a valid evaluation metric and no class weighting or resampling is needed.

**Table 1: Feature Type Classification**

| Feature | Type | Description |
|---------|------|-------------|
| `unique_id` | Identifier | Student grouping key (not a predictive feature) |
| `Painting` | Label | Target variable (3 classes) |
| Emotion intensity (1--10) | Numerical | Continuous rating of emotional intensity |
| Colours count | Numerical | Number of prominent colours perceived |
| Objects count | Numerical | Number of objects noticed |
| Sombre / Content / Calm / Uneasy | Ordinal (Likert 1--5) | Stored as strings, e.g., "4 - Agree" |
| Price willing to pay | Text (numeric extraction needed) | Free-text dollar amounts with inconsistent formatting |
| Room / Who / Season | Multi-select categorical | Comma-separated option lists |
| Describe feeling | Free text | Open-ended emotional response |
| Food analogy | Free text | Creative food comparison |
| Soundtrack description | Free text | Open-ended soundtrack description |

**Numerical features.** Emotion intensity has mean 6.3 (SD = 2.2) with mild left skew, indicating students generally perceived moderate-to-strong emotions. Colours count (mean = 3.9, SD = 3.5) and objects count (mean = 4.0, SD = 3.8) both exhibit strong right skew (skewness > 11) driven by outliers reporting values up to 100 (Figure 2).

**Likert features.** After extracting the numeric prefix from ordinal strings (e.g., "4 - Agree" to 4), the four Likert emotions reveal clear per-class profiles (Figure 3). *The Water Lily Pond* scores highest on calm (4.47) and content (4.41) but lowest on sombre (1.64) and uneasy (1.36). Conversely, *The Persistence of Memory* scores highest on sombre (3.91) and uneasy (3.85). *The Starry Night* falls between the two, leaning toward positive emotions. These Likert features appear strongly discriminative and are likely among the most predictive features.

**Multi-select categoricals.** Room, who, and season columns contain comma-separated selections from fixed option sets (5 room options, 5 companion options, 4 season options). Response patterns vary across classes---for instance, season preferences differ notably, with *The Water Lily Pond* associated more with spring/summer and *The Persistence of Memory* with fall (Figure 4).

**Free-text features.** The three text columns (*describe feeling*, *food analogy*, *soundtrack*) vary in length. Feeling descriptions average 13.7 words (median 8), soundtrack descriptions average 11.6 words, and food analogies are shorter at 3.4 words. The combined vocabulary across text columns is approximately 3,300--3,500 unique words per field (Figure 5). Top words per class show thematic differences: *The Water Lily Pond* elicits words like "calm", "peaceful", "serene", while *The Persistence of Memory* elicits "time", "dread", "melting" (Figure 6).

## 2.2 Data Issues

**Missing values.** All 14 feature columns have missing values, ranging from 65 (3.9%) for emotion intensity to 85 (5.0%) for soundtrack descriptions (Figure 7). Missingness is concentrated: 524 of 562 students (93.2%) have no missing values at all, while 24 students have all three of their rows affected, suggesting these students submitted incomplete surveys. Because missingness is student-level rather than random, we impute using per-class medians for numerical/Likert features and empty strings for text features, preserving class-conditional distributions.

**Outliers.** Using the IQR method, we identified 52 outliers in colours count and 54 in objects count (Figure 8). These are values exceeding 8 and 9.5 respectively---likely erroneous responses (e.g., a student reporting 100 colours). We clip these at the 95th percentile to reduce their influence without discarding rows. Emotion intensity has only 1 outlier and requires no special handling.

**Price column inconsistencies.** The price column is stored as free text with highly variable formatting: "$5", "300 dollars.", "0", "100000000", and non-numeric responses like "a" or "I wouldn't pay much, maybe $10." We extracted the first numeric value via regex, successfully parsing 1,598 of 1,611 non-null entries (99.2%). The resulting distribution is extremely right-skewed (range: \$0 to absurd values exceeding \$10^{15}), with a median of \$100 and 99th percentile at \$401.5M. We apply log(1 + price) transformation and clip at the 99th percentile (Figure 9). Thirteen unparseable entries are treated as missing.

**Text inconsistencies.** One soundtrack response contains HTML tags. 306 feeling descriptions have fewer than 3 words (potentially low-effort responses). We retain these short responses as they may still carry signal, but note their presence. No duplicate text responses were found across different students.

## 2.3 Preprocessing Plan

Each transformation below is motivated by a specific observation from our exploration:

**Likert columns (4 features).** Extract the numeric prefix (1--5) from ordinal strings. Since Likert scores are already on a consistent 1--5 scale with approximately symmetric distributions across classes, no further scaling is needed. *Motivation:* These features showed the strongest per-class separation in our analysis.

**Numerical features (3 features).** Apply standard scaling (z-score normalization) to emotion intensity, colours count, and objects count. *Motivation:* The strong right skew in colours/objects counts (skewness > 11) means raw values would dominate distance-based models; scaling ensures comparable feature magnitudes.

**Price (1 feature).** Clean via regex extraction, clip at 99th percentile, apply log(1 + x) transformation, then standard scale. *Motivation:* The 15-order-of-magnitude range in raw prices makes this feature unusable without transformation; the log transform reduces skew to near-normal.

**Multi-select categoricals (3 features -> ~14 binary features).** Multi-hot encode room (5 options), who (5 options), and season (4 options) into binary indicator columns. *Motivation:* These are multi-label fields where a student can select multiple options; one-hot encoding would create an exponential number of combination categories.

**Free-text features (3 features -> TF-IDF vectors).** Apply TF-IDF vectorization with `max_features=5000`, `min_df=2`, `max_df=0.95`, `ngram_range=(1,2)`, `sublinear_tf=True`, and English stop word removal. *Motivation:* Vocabulary sizes of ~3,300--3,500 words per field are manageable; bigrams capture discriminative phrases (e.g., "makes me feel calm"); `sublinear_tf` dampens high-frequency terms; `min_df=2` removes hapax legomena that add noise.

## 2.4 Data Splitting and Leakage Prevention

Each student contributed exactly **3 survey responses** (one per painting), creating statistical dependence between rows from the same student. A student who consistently uses elaborate language or rates emotions highly would introduce correlated signal across all three of their responses. If these responses were split across train and test sets, the model could implicitly "recognize" the student's style rather than learning painting-specific features, constituting **data leakage**.

We use **`GroupShuffleSplit`** with `unique_id` as the grouping key to ensure all 3 rows from each student remain in the same split. The split ratios are approximately **60/20/20** (train/validation/test):

| Split | Students | Rows | Percentage |
|-------|----------|------|------------|
| Train | 337 | 1,011 | 60.0% |
| Validation | 112 | 336 | 19.9% |
| Test | 113 | 339 | 20.1% |

**Leakage verification:** We assert-verified that no student ID appears in more than one split (intersection of student sets across all split pairs is empty). Class balance is maintained at exactly 33.3% per class in every split, since each student contributes one row per class.

**Important:** The test set was **not used** during any exploration, visualization, or preprocessing parameter selection. All analysis in this section was conducted on the full dataset before splitting, which is acceptable for descriptive exploration (computing statistics and identifying data quality issues) as opposed to model selection.

## 2.5 Key Insights and Model Connections

| Observation | Model Implication |
|-------------|-------------------|
| Perfectly balanced classes (33.3% each) | Accuracy is a valid metric; no class weighting or oversampling needed |
| Likert features show strong per-class separation (e.g., calm ranges from 1.36 to 4.47 across classes) | Simple models (logistic regression, naive Bayes) may perform well on structured features alone |
| High-dimensional sparse TF-IDF text features (~5,000 per text column) | Linear SVM and logistic regression are effective in high-dimensional sparse spaces |
| Small dataset (1,686 rows, 562 students) | Strong regularization needed; avoid high-capacity models (deep networks) that overfit on small data |
| Mixed feature types (numerical, ordinal, binary, sparse text) | Tree-based ensembles (Random Forest, Gradient Boosted Trees) natively handle mixed types without extensive preprocessing |
| `pred.py` constraint (only numpy/pandas allowed at inference) | Must export model weights; logistic regression and naive Bayes are easiest to implement from scratch; tree ensembles require serialized decision rules |
| Strong Likert correlations (content-calm: r=0.69, sombre-uneasy: r=0.66) | Positive/negative emotion pairs are highly correlated---dimensionality reduction (PCA) on Likert features may help, or models robust to multicollinearity (trees, regularized regression) are preferred |
| Price uncorrelated with all other features (|r| < 0.05) | Price carries independent information but is noisy; useful as a supplementary feature after cleaning |

The most discriminative features appear to be the **Likert emotion scores**, which show clear class separation without any transformation beyond numeric extraction. **Text features** provide complementary signal through class-specific vocabulary. We recommend starting with a **regularized logistic regression** on combined TF-IDF + numerical/Likert features as a strong baseline, with **Random Forest** as a comparison model that handles mixed types directly.
