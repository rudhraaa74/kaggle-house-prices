# Kaggle House Prices — Two-Phase AI Coding Prompt

Paste this entire prompt to your code editor's AI tool. All 4 competition files must be in the same directory.

---

## Context

This is for the Kaggle competition: **House Prices: Advanced Regression Techniques**.

Files in this directory:
- `train.csv` — training data with `SalePrice` as the target
- `test.csv` — test data without `SalePrice`
- `sample_submission.csv` — required submission format
- `data_description.txt` — describes every single feature

The competition metric is **RMSLE** (Root Mean Squared Log Error). Goal is the best possible leaderboard score.

---

## Phase 1 — Data Audit (Do This First, Before Writing Any ML Code)

Read all 4 files thoroughly. Then produce a written audit covering every point below. Do not skip any section. Do not write any modeling code yet.

### 1.1 Target Variable
- Plot the distribution of `SalePrice`
- Is it skewed? By how much? What transformation is appropriate?
- What is the range, mean, median?

### 1.2 Missing Values
- List every column with missing values and its missing %
- For each one, cross-reference `data_description.txt` and determine: is this NA semantically meaningful (e.g. "no garage") or is it genuinely missing data?
- Decide the correct imputation strategy for each column based on this — do not use a blanket strategy

### 1.3 Feature Types
- Go through every column and classify it as:
  - **Numeric continuous** (e.g. area, price)
  - **Numeric discrete** (e.g. number of rooms)
  - **Ordinal categorical** (e.g. quality ratings with a natural order)
  - **Nominal categorical** (e.g. neighborhood, house style — no natural order)
  - **Datetime-derived** (e.g. year built, year sold)
- Use `data_description.txt` as the source of truth — do not guess from column names alone

### 1.4 Outliers
- Identify outliers in numeric features, especially in relation to `SalePrice`
- Are there any data points that look like data entry errors or extreme anomalies?
- Decide which (if any) should be removed from training — and justify why

### 1.5 Correlations & Feature Importance
- Which features correlate most strongly with `SalePrice`?
- Are any features highly correlated with each other (multicollinearity)?
- Are there low-variance or near-zero-variance features that are unlikely to help?

### 1.6 Feature Engineering Opportunities
- Based on the data and description, what new features could be derived that would likely improve signal?
- Think about: combining related columns, age calculations, interaction terms, binary flags
- List your ideas with a short justification for each — you will implement these in Phase 2

### 1.7 Model Selection Reasoning
Based on everything you found above, answer:
- How many features will there be after encoding? Is dimensionality a concern?
- Is the data linear enough for regularised regression to work well, or does it need tree-based models?
- What is the likely best model family for this dataset and why?
- Would ensembling/stacking help here? Why or why not?
- What CV strategy is appropriate given the dataset size?

**End Phase 1 with a short summary of your findings and your proposed plan for Phase 2.**

---

## Phase 2 — Build the Pipeline (Only After Phase 1 Is Complete)

Now implement everything, justified by your Phase 1 findings. Produce:

### Deliverable 1: `solution.ipynb`
A clean Jupyter notebook with markdown cells explaining every decision. Structure:

1. **Imports & Setup** — install any missing packages, set seeds for reproducibility
2. **Load Data** — load all files, print shapes and dtypes
3. **EDA Visualisations** — key plots from Phase 1 (target distribution, missing value chart, correlation heatmap, top feature relationships). Keep it focused — only plots that inform decisions
4. **Preprocessing** — implement exactly what Phase 1 concluded:
   - Semantic NA handling (column by column, not blanket)
   - Feature engineering (only the ideas you justified in Phase 1)
   - Correct encoding per feature type (ordinal mapped to integers in the right order, nominal one-hot encoded)
   - Scaling appropriate to the chosen models
   - Use `sklearn Pipeline` and `ColumnTransformer` so train/test are processed identically with no leakage
5. **Modelling**:
   - Choose models based on Phase 1 reasoning — do not train models you don't believe will help
   - Cross-validate each model (5-fold, RMSE on log-transformed target)
   - Print a comparison table of CV scores
   - Tune hyperparameters based on CV results, not guesswork
6. **Final Model**:
   - If stacking/blending is justified, implement it; otherwise use the best single model
   - Report final CV RMSE
7. **Submission**:
   - Predict on `test.csv`
   - Reverse any target transformation
   - Save as `submission.csv` matching the format in `sample_submission.csv`
   - Print first 5 rows of submission as a sanity check

### Deliverable 2: `submission.csv`
Ready to upload directly to Kaggle.

---

## Hard Rules

- **Never apply the same NA strategy to all columns blindly** — each column's treatment must come from `data_description.txt`
- **Never one-hot encode ordinal features** — preserve their order
- **No data leakage** — all preprocessing fit on train only, applied to test
- **No hardcoded magic numbers** — if you choose a hyperparameter value, explain why in a comment or markdown cell
- **Every non-obvious decision needs a one-line justification** in the code or a markdown cell above it

---

## Final Output

At the very end of the notebook, print a summary cell:
- Final model architecture
- CV RMSE score
- Key features that mattered most (feature importances or coefficients if available)
- Any caveats or things you'd try next to improve the score