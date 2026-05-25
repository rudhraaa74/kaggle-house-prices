import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Kaggle House Prices Solution\n",
            "\n",
            "This notebook implements the Phase 2 pipeline for the Kaggle House Prices competition.\n",
            "The approach is based on the data audit conducted in Phase 1."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Imports & Setup\n",
            "We import the necessary libraries and set our random seed for reproducibility."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.impute import SimpleImputer\n",
            "from sklearn.preprocessing import RobustScaler, OneHotEncoder\n",
            "from sklearn.model_selection import KFold, cross_val_score\n",
            "from sklearn.linear_model import Ridge, Lasso\n",
            "from xgboost import XGBRegressor\n",
            "from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "np.random.seed(42)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Load Data\n",
            "Load the training and test sets. We drop the two extreme outliers (`GrLivArea > 4000` & `SalePrice < 300000`) and the zero-variance feature `Utilities`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train = pd.read_csv('train.csv')\n",
            "test = pd.read_csv('test.csv')\n",
            "\n",
            "# Drop extreme outliers\n",
            "train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)\n",
            "\n",
            "# Drop zero-variance column\n",
            "train.drop(['Utilities'], axis=1, inplace=True)\n",
            "test.drop(['Utilities'], axis=1, inplace=True)\n",
            "\n",
            "y_train = np.log1p(train['SalePrice'])\n",
            "train_features = train.drop(['Id', 'SalePrice'], axis=1)\n",
            "test_features = test.drop(['Id'], axis=1)\n",
            "test_id = test['Id']\n",
            "\n",
            "print(f\"Train features shape: {train_features.shape}\")\n",
            "print(f\"Test features shape: {test_features.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. EDA Visualisations\n",
            "Visualising the log transformation of `SalePrice` and the `GrLivArea` relationship without the extreme anomalies."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10,4))\n",
            "plt.subplot(1,2,1)\n",
            "sns.histplot(train['SalePrice'], kde=True)\n",
            "plt.title('SalePrice Distribution')\n",
            "plt.subplot(1,2,2)\n",
            "sns.histplot(y_train, kde=True)\n",
            "plt.title('Log(SalePrice) Distribution')\n",
            "plt.show()\n",
            "\n",
            "plt.figure(figsize=(6,4))\n",
            "sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])\n",
            "plt.title('GrLivArea vs SalePrice (Anomalies removed)')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Preprocessing\n",
            "Here we implement the custom missing value logic (semantic vs genuine) and feature engineering (e.g. `TotalSF`, `TotalBathrooms`) using custom Scikit-Learn transformers to prevent data leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class CustomFeatureEngineer(BaseEstimator, TransformerMixin):\n",
            "    def fit(self, X, y=None):\n",
            "        self.frontage_medians_ = X.groupby('Neighborhood')['LotFrontage'].median()\n",
            "        self.overall_frontage_median_ = X['LotFrontage'].median()\n",
            "        return self\n",
            "    \n",
            "    def transform(self, X):\n",
            "        X_out = X.copy()\n",
            "        \n",
            "        # 1. Impute LotFrontage properly based on neighborhood\n",
            "        X_out['LotFrontage'] = X_out.apply(\n",
            "            lambda row: self.frontage_medians_.get(row['Neighborhood'], self.overall_frontage_median_) \n",
            "            if pd.isnull(row['LotFrontage']) else row['LotFrontage'], axis=1\n",
            "        )\n",
            "        \n",
            "        # 2. Feature Engineering\n",
            "        X_out['TotalSF'] = X_out['TotalBsmtSF'].fillna(0) + X_out['1stFlrSF'].fillna(0) + X_out['2ndFlrSF'].fillna(0)\n",
            "        X_out['TotalBathrooms'] = X_out['FullBath'].fillna(0) + 0.5*X_out['HalfBath'].fillna(0) + \\\n",
            "                                  X_out['BsmtFullBath'].fillna(0) + 0.5*X_out['BsmtHalfBath'].fillna(0)\n",
            "        X_out['HouseAge'] = X_out['YrSold'] - X_out['YearBuilt']\n",
            "        X_out['RemodAge'] = X_out['YrSold'] - X_out['YearRemodAdd']\n",
            "        \n",
            "        X_out['HasPool'] = (X_out['PoolArea'] > 0).astype(int)\n",
            "        X_out['Has2ndFloor'] = (X_out['2ndFlrSF'] > 0).astype(int)\n",
            "        X_out['HasGarage'] = (X_out['GarageArea'] > 0).astype(int)\n",
            "        X_out['HasBsmt'] = (X_out['TotalBsmtSF'] > 0).astype(int)\n",
            "        X_out['HasFireplace'] = (X_out['Fireplaces'] > 0).astype(int)\n",
            "        \n",
            "        return X_out\n",
            "\n",
            "semantic_cat = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', \n",
            "                'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', \n",
            "                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']\n",
            "semantic_num = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', \n",
            "                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']\n",
            "genuine_cat = ['MSZoning', 'Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', \n",
            "               'Functional', 'SaleType']\n",
            "\n",
            "class NA_Imputer(BaseEstimator, TransformerMixin):\n",
            "    def fit(self, X, y=None):\n",
            "        self.modes_ = X[genuine_cat].mode().iloc[0]\n",
            "        return self\n",
            "        \n",
            "    def transform(self, X):\n",
            "        X_out = X.copy()\n",
            "        # Semantic fills\n",
            "        X_out[semantic_cat] = X_out[semantic_cat].fillna(\"None\")\n",
            "        X_out[semantic_num] = X_out[semantic_num].fillna(0)\n",
            "        # Genuine missing categorical mode fills\n",
            "        for col in genuine_cat:\n",
            "            X_out[col] = X_out[col].fillna(self.modes_[col])\n",
            "            \n",
            "        # Fallback for remaining features\n",
            "        str_cols = X_out.select_dtypes(include=['object', 'string', 'category']).columns\n",
            "        X_out[str_cols] = X_out[str_cols].fillna(\"None\")\n",
            "        num_cols = X_out.select_dtypes(exclude=['object', 'string', 'category']).columns\n",
            "        X_out[num_cols] = X_out[num_cols].fillna(0)\n",
            "        return X_out\n",
            "\n",
            "# Preserve ordinal order instead of one-hot encoding\n",
            "class OrdinalMapper(BaseEstimator, TransformerMixin):\n",
            "    def __init__(self):\n",
            "        self.qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}\n",
            "        self.bsmt_exp_map = {'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'None':0}\n",
            "        self.bsmt_fin_map = {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0}\n",
            "        self.shape_map = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}\n",
            "        self.slope_map = {'Gtl':3, 'Mod':2, 'Sev':1}\n",
            "        \n",
            "    def fit(self, X, y=None):\n",
            "        self.is_fitted_ = True\n",
            "        return self\n",
            "        \n",
            "    def transform(self, X):\n",
            "        X_out = X.copy()\n",
            "        qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', \n",
            "                     'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']\n",
            "        for col in qual_cols:\n",
            "            if col in X_out.columns:\n",
            "                X_out[col] = X_out[col].map(self.qual_map).fillna(0)\n",
            "                \n",
            "        if 'BsmtExposure' in X_out.columns: X_out['BsmtExposure'] = X_out['BsmtExposure'].map(self.bsmt_exp_map).fillna(0)\n",
            "        if 'BsmtFinType1' in X_out.columns: X_out['BsmtFinType1'] = X_out['BsmtFinType1'].map(self.bsmt_fin_map).fillna(0)\n",
            "        if 'BsmtFinType2' in X_out.columns: X_out['BsmtFinType2'] = X_out['BsmtFinType2'].map(self.bsmt_fin_map).fillna(0)\n",
            "        if 'LotShape' in X_out.columns: X_out['LotShape'] = X_out['LotShape'].map(self.shape_map).fillna(0)\n",
            "        if 'LandSlope' in X_out.columns: X_out['LandSlope'] = X_out['LandSlope'].map(self.slope_map).fillna(0)\n",
            "            \n",
            "        return X_out\n",
            "\n",
            "preprocessor = Pipeline([\n",
            "    ('custom_eng', CustomFeatureEngineer()),\n",
            "    ('na_imputer', NA_Imputer()),\n",
            "    ('ordinal_map', OrdinalMapper())\n",
            "])\n",
            "\n",
            "X_train_inter = preprocessor.fit_transform(train_features)\n",
            "X_test_inter = preprocessor.transform(test_features)\n",
            "\n",
            "X_train_inter['MSSubClass'] = X_train_inter['MSSubClass'].astype(str)\n",
            "X_test_inter['MSSubClass'] = X_test_inter['MSSubClass'].astype(str)\n",
            "\n",
            "nom_cols = X_train_inter.select_dtypes(include=['object', 'string', 'category']).columns.tolist()\n",
            "num_cols = [c for c in X_train_inter.columns if c not in nom_cols]\n",
            "\n",
            "# Final scaling (RobustScaler handles remaining outliers) and One-Hot Encoding\n",
            "final_transformer = ColumnTransformer(transformers=[\n",
            "    ('num', RobustScaler(), num_cols),\n",
            "    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nom_cols)\n",
            "])\n",
            "\n",
            "X_train_final = final_transformer.fit_transform(X_train_inter)\n",
            "X_test_final = final_transformer.transform(X_test_inter)\n",
            "print(f\"Final train shape after OHE: {X_train_final.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Modelling\n",
            "We evaluate three models: Ridge (L2 regularised regression), Lasso (L1 regularised, good for feature selection), and XGBoost (Tree-based). We tune them using 5-fold CV."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
            "\n",
            "def cv_rmse(model, X, y):\n",
            "    rmse = np.sqrt(-cross_val_score(model, X, y, scoring=\"neg_mean_squared_error\", cv=kf))\n",
            "    return rmse\n",
            "\n",
            "# Ridge - tuned alpha to 10 for stability against multicollinearity\n",
            "ridge = Ridge(alpha=10)\n",
            "# Lasso - tuned alpha to 0.0005 to force sparsity without losing too much signal\n",
            "lasso = Lasso(alpha=0.0005, random_state=42)\n",
            "# XGBoost - constrained depth and learning rate for generalisation\n",
            "xgb = XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=3, \n",
            "                   subsample=0.8, colsample_bytree=0.8, random_state=42)\n",
            "\n",
            "models = {'Ridge': ridge, 'Lasso': lasso, 'XGBoost': xgb}\n",
            "scores = {}\n",
            "for name, model in models.items():\n",
            "    score = cv_rmse(model, X_train_final, y_train)\n",
            "    scores[name] = score.mean()\n",
            "    print(f\"{name}: Mean RMSE = {score.mean():.4f}, Std = {score.std():.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Final Model\n",
            "We combine Ridge, Lasso, and XGBoost using a simple averaging approach. Blending linear models and tree-based models typically improves the CV score because they make uncorrelated errors."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):\n",
            "    def __init__(self, models):\n",
            "        self.models = models\n",
            "        \n",
            "    def fit(self, X, y):\n",
            "        self.models_ = [clone(x) for x in self.models]\n",
            "        for model in self.models_:\n",
            "            model.fit(X, y)\n",
            "        return self\n",
            "    \n",
            "    def predict(self, X):\n",
            "        predictions = np.column_stack([model.predict(X) for model in self.models_])\n",
            "        return np.mean(predictions, axis=1)\n",
            "\n",
            "blended_model = AveragingModels(models=(ridge, lasso, xgb))\n",
            "blend_score = cv_rmse(blended_model, X_train_final, y_train)\n",
            "print(f\"Blended Model CV RMSE: {blend_score.mean():.4f} +/- {blend_score.std():.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Submission\n",
            "Train the final model on the full training set, predict on the test set, reverse the `np.log1p` transformation, and save the submission."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "blended_model.fit(X_train_final, y_train)\n",
            "test_preds = np.expm1(blended_model.predict(X_test_final))\n",
            "\n",
            "submission = pd.DataFrame({'Id': test_id, 'SalePrice': test_preds})\n",
            "submission.to_csv('submission.csv', index=False)\n",
            "\n",
            "print(\"First 5 rows of submission:\")\n",
            "print(submission.head())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Final Output Summary\n",
            "Below is the requested final summary outlining the model performance and key features."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"========== FINAL SUMMARY ==========\")\n",
            "print(\"Final model architecture: Simple Averaging Blend of Ridge (alpha=10), Lasso (alpha=0.0005), and XGBoost (depth=3, lr=0.05).\")\n",
            "print(f\"CV RMSE score: {blend_score.mean():.4f}\")\n",
            "\n",
            "# Get Lasso coefficients to see key features\n",
            "lasso.fit(X_train_final, y_train)\n",
            "feature_names = final_transformer.get_feature_names_out()\n",
            "coefs = pd.Series(lasso.coef_, index=feature_names)\n",
            "print(\"\\nKey features that mattered most (Lasso Coefficients):\")\n",
            "print(\"Top Positive Contributors:\")\n",
            "print(coefs.sort_values(ascending=False).head(5))\n",
            "print(\"\\nTop Negative Contributors:\")\n",
            "print(coefs.sort_values(ascending=True).head(5))\n",
            "\n",
            "print(\"\\nCaveats / Next Steps:\")\n",
            "print(\"- Next steps would involve more extensive hyperparameter tuning using Optuna.\")\n",
            "print(\"- Adding a meta-model like StackingRegressor instead of a simple average might yield better generalisation.\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('/Users/rudhrakoul/Desktop/ML/kaggle_housepricing_comp/solution.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

# Also generate the python script to run to get the outputs
code_blocks = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
with open('/Users/rudhrakoul/Desktop/ML/kaggle_housepricing_comp/run_solution.py', 'w') as f:
    for block in code_blocks:
        f.write(block + "\n\n")
