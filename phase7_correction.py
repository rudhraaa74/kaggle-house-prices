import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PowerTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from category_encoders import TargetEncoder
from sklearn.impute import KNNImputer
from scipy.stats import skew
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- LOAD DATA ---
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# NOTE: Removed SaleCondition == 'Normal' filter to prevent train/test distribution mismatch

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
        self.is_fitted_ = True
        return self
    def transform(self, X):
        X_out = X.copy()
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
genuine_num = ['LotFrontage']

class NA_Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.modes_ = X[genuine_cat].mode().iloc[0]
        self.medians_ = X[genuine_num].median() if len(genuine_num) > 0 else pd.Series()
        self.is_fitted_ = True
        return self
    def transform(self, X):
        X_out = X.copy()
        X_out[semantic_cat] = X_out[semantic_cat].fillna("None")
        X_out[semantic_num] = X_out[semantic_num].fillna(0)
        return X_out

class AdvancedKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, num_cols):
        self.knn = KNNImputer(n_neighbors=5)
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
    def fit(self, X, y=None):
        self.encoders = {}
        X_tmp = X.copy()
        self.str_cols = X_tmp.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        for c in self.str_cols:
            unique_vals = X_tmp[c].dropna().unique()
            self.encoders[c] = {v: i for i, v in enumerate(unique_vals)}
            X_tmp[c] = X_tmp[c].map(self.encoders[c])
        self.knn.fit(X_tmp)
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        X_tmp = X.copy()
        for c in self.str_cols:
            if c in X_tmp.columns:
                X_tmp[c] = X_tmp[c].map(self.encoders.get(c, {}))
        X_imp = pd.DataFrame(self.knn.transform(X_tmp), columns=X_tmp.columns, index=X_tmp.index)
        for c in self.str_cols:
            if c in self.cat_cols:
                X_imp[c] = X_imp[c].round()
                rev_map = {i: v for v, i in self.encoders.get(c, {}).items()}
                X_imp[c] = X_imp[c].map(rev_map)
                
        X_out = X.copy()
        for c in self.cat_cols:
            if c in X_out.columns:
                X_out[c] = X_out[c].fillna(X_imp[c])
        for c in self.num_cols:
            if c in X_out.columns:
                X_out[c] = X_out[c].fillna(X_imp[c])
                
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

steps = [
    ('custom_eng', CustomFeatureEngineer()),
    ('na_imputer', NA_Imputer()),
    ('knn_imputer', AdvancedKNNImputer(cat_cols=genuine_cat, num_cols=genuine_num)),
    ('ordinal_map', OrdinalMapper())
]

preprocessor = Pipeline(steps)

X_train_inter = preprocessor.fit_transform(train_features)
X_test_inter = preprocessor.transform(test_features)
X_train_inter['MSSubClass'] = X_train_inter['MSSubClass'].astype(str)
X_test_inter['MSSubClass'] = X_test_inter['MSSubClass'].astype(str)

nom_cols = X_train_inter.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
num_cols = [c for c in X_train_inter.columns if c not in nom_cols]

high_card_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
low_card_cols = [c for c in nom_cols if c not in high_card_cols]

# Skew normalization
skewness = X_train_inter[num_cols].apply(lambda x: skew(x.dropna()))
skewed_cols = skewness[abs(skewness) > 0.75].index.tolist()
normal_num_cols = [c for c in num_cols if c not in skewed_cols]

transformers = [
    ('num_normal', RobustScaler(), normal_num_cols),
    ('num_skewed', Pipeline([
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler())
    ]), skewed_cols),
    ('high_card', TargetEncoder(), high_card_cols),
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cols)
]

final_transformer = ColumnTransformer(transformers=transformers)
feature_selector = SelectFromModel(Lasso(alpha=0.0005, random_state=42))

def create_pipeline(model):
    return Pipeline([
        ('final_transformer', final_transformer),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Highly regularized tree models
xgb_params = {'n_estimators': 1000, 'learning_rate': 0.015, 'max_depth': 3, 'min_child_weight': 10, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_lambda': 10.0}
lgbm_params = {'n_estimators': 1000, 'learning_rate': 0.015, 'max_depth': 3, 'min_child_samples': 30, 'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_lambda': 10.0}

ridge_model = create_pipeline(Ridge(alpha=10))
lasso_model = create_pipeline(Lasso(alpha=0.0005, random_state=42))
xgb_model = create_pipeline(XGBRegressor(**xgb_params, random_state=42, n_jobs=1))
lgbm_model = create_pipeline(LGBMRegressor(**lgbm_params, random_state=42, n_jobs=1, verbose=-1))

stack_estimators = [
    ('ridge', ridge_model),
    ('lasso', lasso_model),
    ('xgb', xgb_model),
    ('lgbm', lgbm_model)
]

stacked_model = StackingRegressor(
    estimators=stack_estimators,
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
    cv=kf,
    passthrough=False,
    n_jobs=1
)

print("Running Phase 7 (Corrected Train Set + KNN + Yeo-Johnson) ...", flush=True)
rmse = np.sqrt(-cross_val_score(stacked_model, X_train_inter, y_train, scoring="neg_mean_squared_error", cv=kf, n_jobs=1)).mean()

print("\n=== Phase 7 CV Results ===")
print(f"Corrected Realistic CV RMSE: {rmse:.5f}")

# Train and Predict
stacked_model.fit(X_train_inter, y_train)
final_preds = np.expm1(stacked_model.predict(X_test_inter))
submission = pd.DataFrame({'Id': test_id, 'SalePrice': final_preds})
submission.to_csv('submission.csv', index=False)
print("submission.csv generated!")

# Append to solution.ipynb
cells_to_append = []
code_final = f"""# Phase 7: Distribution Mismatch Correction
# We reverted the SaleCondition=='Normal' filter to prevent Train/Test mismatch.
# We retained the advanced KNNImputer and the Yeo-Johnson PowerTransformer.
#
# Corrected Realistic CV RMSE: {rmse:.5f}

# (The submission.csv has been updated with the corrected architecture trained on the full dataset.)
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["## Phase 7: Distribution Mismatch Correction\n", "We removed the truncation of non-Normal SaleConditions to ensure the training set distribution matches the test set. The `KNNImputer` and `PowerTransformer` were retained for generalized robustness."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_final.split("\n")]})

with open('solution.ipynb', 'r') as f:
    nb = json.load(f)

nb['cells'].extend(cells_to_append)
with open('solution.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
