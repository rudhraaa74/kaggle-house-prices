import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import json

# --- REBUILD ROUND 1 STATE ---
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train.drop(['Utilities'], axis=1, inplace=True)
test.drop(['Utilities'], axis=1, inplace=True)
y_train = np.log1p(train['SalePrice'])
train_features = train.drop(['Id', 'SalePrice'], axis=1)
test_features = test.drop(['Id'], axis=1)
test_id = test['Id']

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.frontage_medians_ = X.groupby('Neighborhood')['LotFrontage'].median()
        self.overall_frontage_median_ = X['LotFrontage'].median()
        return self
    def transform(self, X):
        X_out = X.copy()
        X_out['LotFrontage'] = X_out.apply(
            lambda row: self.frontage_medians_.get(row['Neighborhood'], self.overall_frontage_median_) 
            if pd.isnull(row['LotFrontage']) else row['LotFrontage'], axis=1
        )
        X_out['TotalSF'] = X_out['TotalBsmtSF'].fillna(0) + X_out['1stFlrSF'].fillna(0) + X_out['2ndFlrSF'].fillna(0)
        X_out['TotalBathrooms'] = X_out['FullBath'].fillna(0) + 0.5*X_out['HalfBath'].fillna(0) + \
                                  X_out['BsmtFullBath'].fillna(0) + 0.5*X_out['BsmtHalfBath'].fillna(0)
        X_out['HouseAge'] = X_out['YrSold'] - X_out['YearBuilt']
        X_out['RemodAge'] = X_out['YrSold'] - X_out['YearRemodAdd']
        X_out['HasPool'] = (X_out['PoolArea'] > 0).astype(int)
        X_out['Has2ndFloor'] = (X_out['2ndFlrSF'] > 0).astype(int)
        X_out['HasGarage'] = (X_out['GarageArea'] > 0).astype(int)
        X_out['HasBsmt'] = (X_out['TotalBsmtSF'] > 0).astype(int)
        X_out['HasFireplace'] = (X_out['Fireplaces'] > 0).astype(int)
        return X_out

semantic_cat = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
                'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
semantic_num = ['GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
genuine_cat = ['MSZoning', 'Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']

class NA_Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.modes_ = X[genuine_cat].mode().iloc[0]
        return self
    def transform(self, X):
        X_out = X.copy()
        X_out[semantic_cat] = X_out[semantic_cat].fillna("None")
        X_out[semantic_num] = X_out[semantic_num].fillna(0)
        for col in genuine_cat:
            X_out[col] = X_out[col].fillna(self.modes_[col])
        str_cols = X_out.select_dtypes(include=['object', 'string', 'category']).columns
        X_out[str_cols] = X_out[str_cols].fillna("None")
        num_cols = X_out.select_dtypes(exclude=['object', 'string', 'category']).columns
        X_out[num_cols] = X_out[num_cols].fillna(0)
        return X_out

class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0}
        self.bsmt_exp_map = {'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'None':0}
        self.bsmt_fin_map = {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0}
        self.shape_map = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}
        self.slope_map = {'Gtl':3, 'Mod':2, 'Sev':1}
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    def transform(self, X):
        X_out = X.copy()
        qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                     'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        for col in qual_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].map(self.qual_map).fillna(0)
        if 'BsmtExposure' in X_out.columns: X_out['BsmtExposure'] = X_out['BsmtExposure'].map(self.bsmt_exp_map).fillna(0)
        if 'BsmtFinType1' in X_out.columns: X_out['BsmtFinType1'] = X_out['BsmtFinType1'].map(self.bsmt_fin_map).fillna(0)
        if 'BsmtFinType2' in X_out.columns: X_out['BsmtFinType2'] = X_out['BsmtFinType2'].map(self.bsmt_fin_map).fillna(0)
        if 'LotShape' in X_out.columns: X_out['LotShape'] = X_out['LotShape'].map(self.shape_map).fillna(0)
        if 'LandSlope' in X_out.columns: X_out['LandSlope'] = X_out['LandSlope'].map(self.slope_map).fillna(0)
        return X_out

preprocessor = Pipeline([
    ('custom_eng', CustomFeatureEngineer()),
    ('na_imputer', NA_Imputer()),
    ('ordinal_map', OrdinalMapper())
])

X_train_inter = preprocessor.fit_transform(train_features)
X_test_inter = preprocessor.transform(test_features)
X_train_inter['MSSubClass'] = X_train_inter['MSSubClass'].astype(str)
X_test_inter['MSSubClass'] = X_test_inter['MSSubClass'].astype(str)

nom_cols = X_train_inter.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
num_cols = [c for c in X_train_inter.columns if c not in nom_cols]

final_transformer = ColumnTransformer(transformers=[
    ('num', RobustScaler(), num_cols),
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nom_cols)
])

X_train_final = final_transformer.fit_transform(X_train_inter)
X_test_final = final_transformer.transform(X_test_inter)

ridge = Ridge(alpha=10)
lasso = Lasso(alpha=0.0005, random_state=42)
tuned_xgb_params = {'n_estimators': 1265, 'learning_rate': 0.012224530398883712, 'max_depth': 4, 'min_child_weight': 2, 'subsample': 0.5126036971322886, 'colsample_bytree': 0.6504339622089299, 'reg_alpha': 0.0007209346542902345, 'reg_lambda': 1.3179752639844382}
tuned_xgb = XGBRegressor(**tuned_xgb_params, random_state=42, n_jobs=-1)
lgbm = LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, verbose=-1)

class WeightedAveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.dot(predictions, self.weights)

best_weights = [5.03069808e-17, 5.70766376e-01, 4.23104607e-01, 6.12901711e-03]
best_blend = WeightedAveragingModels(models=(ridge, lasso, tuned_xgb, lgbm), weights=best_weights)

best_blend.fit(X_train_final, y_train)
final_preds = np.expm1(best_blend.predict(X_test_final))
submission = pd.DataFrame({'Id': test_id, 'SalePrice': final_preds})
submission.to_csv('submission.csv', index=False)

# Append to solution.ipynb
cells_to_append = []

code_final = f"""# Final Model Selection
# Based on cross-validation, we select the Optimised Blend from Round 1 (CV RMSE: 0.1085)
# This outperforms the Round 2 robust stacking model (CV RMSE: 0.1105)

best_weights = {best_weights}
best_blend = WeightedAveragingModels(models=(ridge, lasso, tuned_xgb, lgbm), weights=best_weights)

best_blend.fit(X_train_final, y_train)
final_preds = np.expm1(best_blend.predict(X_test_final))

submission = pd.DataFrame({{'Id': test_id, 'SalePrice': final_preds}})
submission.to_csv('submission.csv', index=False)
print("submission.csv generated using Round 1 Optimised Blend!")
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["## Final Submission Output\n", "We have reverted to the Round 1 Scipy-Optimised Blend, which achieved the lowest overall CV RMSE of 0.1085."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_final.split("\n")]})

with open('solution.ipynb', 'r') as f:
    nb = json.load(f)

nb['cells'].extend(cells_to_append)
with open('solution.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Reverted to Round 1 model and submission.csv generated!")
