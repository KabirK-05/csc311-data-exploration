# Data Exploration Reporting Checklist

Use this checklist before finalizing the data exploration report to ensure all rubric
requirements are met. Each item maps to specific marks in the CSC311 grading rubric.

## Section 1: Dataset Summary (0.5 marks)

- [ ] Feature types are explicitly classified (numerical, categorical, text)
- [ ] Distributions are described for each feature type
- [ ] Class balance is reported with exact counts and percentages
- [ ] A class distribution figure is included
- [ ] Summary statistics table is included

## Section 2: Data Issues (0.5 marks)

- [ ] Missing values are checked and counts reported
- [ ] Outliers are identified (with method: IQR, z-score, or visual)
- [ ] Inconsistencies are documented (encoding, duplicates, near-duplicates)
- [ ] For EACH issue: what it is, how many affected, how you handle it

## Section 3: Preprocessing (0.5 marks)

- [ ] Text representation method is stated and justified
- [ ] Normalization/scaling approach is stated and justified
- [ ] Encoding approach for categoricals is stated and justified
- [ ] All transformations are connected to observations from the data

## Section 4: Data Leakage Prevention (0.5 marks)

- [ ] Test set reservation is explicitly described
- [ ] Statement that test set was NOT used during exploration
- [ ] Student grouping constraint is explained (3 points per student must stay together)
- [ ] Grouped splitting method is described (e.g., GroupShuffleSplit)
- [ ] Verification that no student appears in multiple splits
- [ ] Split ratios are stated and justified

## Section 5: Figures (embedded throughout)

- [ ] All figures have titles, axis labels, and legends
- [ ] Each figure has 1-3 sentences of interpretation
- [ ] Figures are referenced in the text (not just appended)
- [ ] Consistent color scheme across figures

## Section 6: Model Choice Connections (part of overall narrative)

- [ ] At least 3 specific observations are connected to model suggestions
- [ ] Connections are concrete, not generic
- [ ] Both linear and nonlinear model families are considered
- [ ] Text feature characteristics inform model choice

## Common Mark Deductions to Avoid

- Missing class balance analysis → lose marks on dataset summary
- No mention of data leakage → lose marks on splitting
- Figures without explanation → figures don't count
- "No missing values found" without checking for implicit missingness → weak analysis
- Generic model suggestions not tied to data → lose marks on connections
