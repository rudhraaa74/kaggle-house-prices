# Kaggle House Prices: Advanced Regression Techniques
## Complete Project Summary

This document outlines the entire end-to-end machine learning pipeline built to predict house prices, from initial data auditing to advanced ensemble optimization.

---

### Phase 1: Data Audit
Before writing any modeling code, we conducted a rigorous audit of the dataset to prevent data leakage and ensure proper preprocessing.
1. **Target Variable:** `SalePrice` was heavily right-skewed. We applied a `log1p` transformation to normalize it and used RMSE on the log scale as our primary metric (which matches Kaggle's RMSLE metric).
2. **Missing Values (Semantic vs. Genuine):**
   - **Semantic Missing:** We identified features where `NA` meant "None" (e.g., `PoolQC` NA = "No Pool", `GarageType` NA = "No Garage"). These were filled with `"None"` (for categorical) or `0` (for numerical).
   - **Genuine Missing:** Features where `NA` truly meant missing data (e.g., `Electrical`, `MSZoning`). These were imputed using the mode.
3. **Outliers:** Based on the dataset documentation, we identified and removed two extreme outliers (`GrLivArea > 4000` & `SalePrice < 300000`) which severely distorted linear models.
4. **Data Types:** Categorical integer columns like `MSSubClass` were explicitly converted to string to prevent the model from treating them as ordinal magnitudes.

---

### Phase 2: Core Pipeline Construction
We constructed a strictly leakage-free Scikit-Learn `Pipeline` and `ColumnTransformer` inside a Jupyter Notebook (`solution.ipynb`).
1. **Custom Transformers:** We built `CustomFeatureEngineer`, `NA_Imputer`, and `OrdinalMapper` classes inheriting from `TransformerMixin`.
2. **Feature Engineering:** We engineered new features like `HouseAge`, `RemodAge`, `TotalSF`, `TotalBathrooms`, and binary indicators like `HasPool`, `HasGarage`, etc.
3. **Ordinal Mapping:** We mapped qualitative scales (e.g., `Ex`, `Gd`, `TA`, `Fa`, `Po`) to integers (5, 4, 3, 2, 1) to preserve their ordinal nature rather than One-Hot Encoding them.
4. **Baseline Modeling:** 
   - We trained `Ridge` (alpha=10), `Lasso` (alpha=0.0005), and an untuned `XGBoost`.
   - A simple unweighted average of these three models achieved a baseline **5-Fold CV RMSE of 0.1100**.

---

### Phase 3: Optimisation Round 1
We expanded the modeling strategy to squeeze out the best performance:
1. **Added LightGBM:** An untuned LightGBM was added but it performed poorly (`0.1271` CV RMSE) and hurt the simple blend.
2. **Scipy Weight Optimisation:** Instead of equal weights, we used `scipy.optimize.minimize` on out-of-fold predictions to mathematically find the optimal ensemble weights.
3. **XGBoost Optuna Tuning:** We ran a 50-trial Optuna study to tune XGBoost, dropping its individual RMSE from `0.1157` to `0.1127`.
4. **Stacking Regressor:** We built a `StackingRegressor` with a Ridge meta-learner, achieving `0.1090`.
5. **Round 1 Result:** The Scipy-Optimised Blend combining Ridge, Lasso, the Tuned XGBoost, and LightGBM achieved our best score: **0.1085 CV RMSE**.

---

### Phase 4: Optimisation Round 2
We attempted a more robust, mathematically pure optimization strategy focusing on strict leakage-free transformations.
1. **Target Encoding:** Replaced One-Hot Encoding for high-cardinality features (`Neighborhood`, `Exterior1st`, `Exterior2nd`) with `category_encoders.TargetEncoder`, strictly embedded inside the cross-validation fold.
2. **Feature Selection:** Added `SelectFromModel(Lasso)` to dynamically strip out noisy, zero-importance features before feeding data to tree models.
3. **LightGBM Tuning:** Tuned LightGBM via Optuna (bringing it down to `0.1166`).
4. **RidgeCV Stacking:** Replaced the hardcoded meta-learner with `RidgeCV` to dynamically find the optimal alpha.
5. **Round 2 Result:** While structurally more robust, the Stacking model achieved **0.1105 CV RMSE**, failing to beat the Round 1 Scipy Blend. 

---

### Final Submission Generation
Because the primary goal is minimizing RMSE for the leaderboard, we **reverted to the Round 1 Optimised Blend** (CV RMSE: 0.1085). 

We retrained this optimal configuration on the full dataset, inverse-transformed (`expm1`) the predictions, completely overwrote `submission.csv`, and appended a final block to `solution.ipynb` to finalize the project.
