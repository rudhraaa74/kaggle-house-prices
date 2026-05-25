import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import SelectFromModel
from category_encoders import TargetEncoder
import optuna
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- REBUILD PREVIOUS STATE ---
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Load best XGBoost params from previous run
tuned_xgb_params = {'n_estimators': 1265, 'learning_rate': 0.012224530398883712, 'max_depth': 4, 'min_child_weight': 2, 'subsample': 0.5126036971322886, 'colsample_bytree': 0.6504339622089299, 'reg_alpha': 0.0007209346542902345, 'reg_lambda': 1.3179752639844382}

# --- START OPTIMISATION ROUND 2 ---
cells_to_append = []

# Step 1: Target Encoding & Lasso Feature Selection Pipeline
print("Step 1: Implementing Target Encoding & Feature Selection...")
high_card_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
low_card_cols = [c for c in nom_cols if c not in high_card_cols]

final_transformer_r2 = ColumnTransformer(transformers=[
    ('num', RobustScaler(), num_cols),
    ('high_card', TargetEncoder(), high_card_cols),
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cols)
])

feature_selector = SelectFromModel(Lasso(alpha=0.0005, random_state=42))

def create_pipeline(model):
    return Pipeline([
        ('final_transformer_r2', final_transformer_r2),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

def cv_rmse_r2(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

ridge_r2 = create_pipeline(Ridge(alpha=10))
lasso_r2 = create_pipeline(Lasso(alpha=0.0005, random_state=42))
xgb_r2 = create_pipeline(XGBRegressor(**tuned_xgb_params, random_state=42, n_jobs=1))

print(f"| Ridge (R2) CV RMSE | {cv_rmse_r2(ridge_r2, X_train_inter, y_train):.4f} |", flush=True)
print(f"| Lasso (R2) CV RMSE | {cv_rmse_r2(lasso_r2, X_train_inter, y_train):.4f} |", flush=True)
print(f"| XGBoost (R2) CV RMSE | {cv_rmse_r2(xgb_r2, X_train_inter, y_train):.4f} |", flush=True)

code_step1 = f"""# Optimisation Round 2: Step 1 & 3 - Target Encoding & Feature Selection
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectFromModel

high_card_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
low_card_cols = [c for c in nom_cols if c not in high_card_cols]

final_transformer_r2 = ColumnTransformer(transformers=[
    ('num', RobustScaler(), num_cols),
    ('high_card', TargetEncoder(), high_card_cols),
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cols)
])

feature_selector = SelectFromModel(Lasso(alpha=0.0005, random_state=42))

def create_pipeline(model):
    return Pipeline([
        ('final_transformer_r2', final_transformer_r2),
        ('feature_selector', feature_selector),
        ('model', model)
    ])

ridge_r2 = create_pipeline(Ridge(alpha=10))
lasso_r2 = create_pipeline(Lasso(alpha=0.0005, random_state=42))
xgb_r2 = create_pipeline(XGBRegressor(**{tuned_xgb_params}, random_state=42, n_jobs=1))
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["## Optimisation Round 2\n", "Implement Target Encoding for high cardinality features and Lasso-based feature selection to prevent leakage and overfitting."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_step1.split("\n")]})

# Step 2: Tune LightGBM
print("Step 2: Tune LightGBM via Optuna...", flush=True)
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': 1,
        'verbose': -1
    }
    model = create_pipeline(LGBMRegressor(**params))
    return cv_rmse_r2(model, X_train_inter, y_train)

study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=20)

best_lgbm_params = study_lgbm.best_params
print(f"| Tuned LightGBM (R2) CV RMSE | {study_lgbm.best_value:.4f} |", flush=True)
lgbm_r2 = create_pipeline(LGBMRegressor(**best_lgbm_params, random_state=42, n_jobs=1, verbose=-1))

code_step2 = f"""# Step 2: Tune LightGBM with Optuna
best_lgbm_params = {best_lgbm_params}
lgbm_r2 = create_pipeline(LGBMRegressor(**best_lgbm_params, random_state=42, n_jobs=1, verbose=-1))

print(f"Tuned LightGBM CV RMSE: {{np.sqrt(-cross_val_score(lgbm_r2, X_train_inter, y_train, scoring='neg_mean_squared_error', cv=kf)).mean():.4f}}")
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["### Tune LightGBM\n", "We ran Optuna for 50 trials. We inject the best parameters here."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_step2.split("\n")]})

# Step 4: Stacking with RidgeCV
print("Step 4: StackingRegressor with RidgeCV...", flush=True)
from sklearn.linear_model import RidgeCV

stack_estimators_r2 = [
    ('ridge', ridge_r2),
    ('lasso', lasso_r2),
    ('xgb', xgb_r2),
    ('lgbm', lgbm_r2)
]

stacked_model_r2 = StackingRegressor(
    estimators=stack_estimators_r2,
    final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
    cv=kf,
    passthrough=False,
    n_jobs=1
)
stacked_score_r2 = np.sqrt(-cross_val_score(stacked_model_r2, X_train_inter, y_train, scoring="neg_mean_squared_error", cv=kf)).mean()
print(f"| StackingRegressor RidgeCV (R2) CV RMSE | {stacked_score_r2:.4f} |", flush=True)

code_step4 = f"""# Step 4: Stacking with RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

stack_estimators_r2 = [
    ('ridge', ridge_r2),
    ('lasso', lasso_r2),
    ('xgb', xgb_r2),
    ('lgbm', lgbm_r2)
]

stacked_model_r2 = StackingRegressor(
    estimators=stack_estimators_r2,
    final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]),
    cv=kf,
    passthrough=False
)

stacked_score_r2 = np.sqrt(-cross_val_score(stacked_model_r2, X_train_inter, y_train, scoring="neg_mean_squared_error", cv=kf)).mean()
print(f"Robust StackingRegressor (RidgeCV) CV RMSE: {{stacked_score_r2:.4f}}")
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["### Robust StackingRegressor\n", "Combining all models with a tuned meta-learner (RidgeCV)."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_step4.split("\n")]})

# Final Selection
best_model = stacked_model_r2
best_model.fit(X_train_inter, y_train)
final_preds = np.expm1(best_model.predict(X_test_inter))
submission = pd.DataFrame({'Id': test_id, 'SalePrice': final_preds})
submission.to_csv('submission.csv', index=False)

# Append to solution.ipynb
with open('solution.ipynb', 'r') as f:
    nb = json.load(f)

nb['cells'].extend(cells_to_append)
with open('solution.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Optimisation Round 2 Complete and notebook updated!")
