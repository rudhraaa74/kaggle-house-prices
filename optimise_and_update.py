import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from scipy.optimize import minimize
import optuna
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load Data and Setup
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_rmse(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

ridge = Ridge(alpha=10)
lasso = Lasso(alpha=0.0005, random_state=42)
xgb = XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=3, 
                   subsample=0.8, colsample_bytree=0.8, random_state=42)

# Opt Round 1

cells_to_append = []

# Step 1: Add LightGBM
lgbm = LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, verbose=-1)
lgbm_score = cv_rmse(lgbm, X_train_final, y_train)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

blended_3 = AveragingModels(models=(ridge, lasso, xgb))
blend_3_score = cv_rmse(blended_3, X_train_final, y_train)

blended_4 = AveragingModels(models=(ridge, lasso, xgb, lgbm))
blend_4_score = cv_rmse(blended_4, X_train_final, y_train)

include_lgbm = blend_4_score < blend_3_score

print(f"| Model | CV RMSE |")
print(f"|---|---|")
print(f"| LightGBM (Untuned) | {lgbm_score:.4f} |")
print(f"| Blend (Ridge, Lasso, XGB) | {blend_3_score:.4f} |")
print(f"| Blend (Ridge, Lasso, XGB, LGBM) | {blend_4_score:.4f} |")
print(f"| LightGBM kept in blend? | {'Yes' if include_lgbm else 'No'} |")

code_step1 = f"""# Step 1: Add LightGBM
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, verbose=-1)
lgbm_score = np.sqrt(-cross_val_score(lgbm, X_train_final, y_train, scoring="neg_mean_squared_error", cv=kf)).mean()
print(f"LightGBM CV RMSE: {{lgbm_score:.4f}}")

blended_4 = AveragingModels(models=(ridge, lasso, xgb, lgbm))
blend_4_score = np.sqrt(-cross_val_score(blended_4, X_train_final, y_train, scoring="neg_mean_squared_error", cv=kf)).mean()
print(f"Blend 4 (with LightGBM) CV RMSE: {{blend_4_score:.4f}}")
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["## Optimisation Step 1: LightGBM\n", "Evaluate LightGBM and add to the simple average blend if it improves the score."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_step1.split("\n")]})

# Step 2: Optimise Blend Weights
oof_ridge = cross_val_predict(ridge, X_train_final, y_train, cv=kf)
oof_lasso = cross_val_predict(lasso, X_train_final, y_train, cv=kf)
oof_xgb = cross_val_predict(xgb, X_train_final, y_train, cv=kf)
oof_lgbm = cross_val_predict(lgbm, X_train_final, y_train, cv=kf)
oof_preds = np.column_stack([oof_ridge, oof_lasso, oof_xgb, oof_lgbm])

def rmse_objective(weights):
    blend_pred = np.dot(oof_preds, weights)
    return np.sqrt(np.mean((y_train - blend_pred)**2))

init_weights = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
res = minimize(rmse_objective, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
opt_blend_score = res.fun

print(f"| Optimised Blend Weights | {res.x} |")
print(f"| Optimised Blend CV RMSE | {opt_blend_score:.4f} |")

code_step2 = f"""# Step 2: Optimise blend weights using out-of-fold predictions
from sklearn.model_selection import cross_val_predict
from scipy.optimize import minimize

oof_ridge = cross_val_predict(ridge, X_train_final, y_train, cv=kf)
oof_lasso = cross_val_predict(lasso, X_train_final, y_train, cv=kf)
oof_xgb = cross_val_predict(xgb, X_train_final, y_train, cv=kf)
oof_lgbm = cross_val_predict(lgbm, X_train_final, y_train, cv=kf)

oof_preds = np.column_stack([oof_ridge, oof_lasso, oof_xgb, oof_lgbm])

def rmse_objective(weights):
    blend_pred = np.dot(oof_preds, weights)
    return np.sqrt(np.mean((y_train - blend_pred)**2))

init_weights = np.ones(4) / 4.0
bounds = [(0, 1)] * 4
cons = ({{'type': 'eq', 'fun': lambda w: 1 - sum(w)}})
res = minimize(rmse_objective, init_weights, method='SLSQP', bounds=bounds, constraints=cons)

print(f"Optimised weights (Ridge, Lasso, XGB, LGBM): {{res.x}}")
print(f"Optimised blend CV RMSE: {{res.fun:.4f}}")
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["## Optimisation Step 2: Optimal Weights\n", "Instead of a simple average, find the optimal weights using Scipy minimize."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_step2.split("\n")]})


# Step 3: Tune XGBoost with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }
    model = XGBRegressor(**params)
    return cv_rmse(model, X_train_final, y_train)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
tuned_xgb_params = study.best_params
tuned_xgb_score = study.best_value

tuned_xgb = XGBRegressor(**tuned_xgb_params, random_state=42, n_jobs=-1)
oof_tuned_xgb = cross_val_predict(tuned_xgb, X_train_final, y_train, cv=kf)
oof_preds_tuned = np.column_stack([oof_preds[:, 0], oof_preds[:, 1], oof_tuned_xgb, oof_preds[:, 3]])
def rmse_objective_tuned(weights):
    blend_pred = np.dot(oof_preds_tuned, weights)
    return np.sqrt(np.mean((y_train - blend_pred)**2))
res_tuned = minimize(rmse_objective_tuned, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
opt_blend_tuned_score = res_tuned.fun

print(f"| Tuned XGBoost CV RMSE | {tuned_xgb_score:.4f} |")
print(f"| Tuned XGBoost Params | {tuned_xgb_params} |")
print(f"| Optimised Blend with Tuned XGB RMSE | {opt_blend_tuned_score:.4f} |")

code_step3 = f"""# Step 3: Tune XGBoost with Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {{
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }}
    model = XGBRegressor(**params)
    return np.sqrt(-cross_val_score(model, X_train_final, y_train, scoring="neg_mean_squared_error", cv=kf)).mean()

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

best_xgb_params = {tuned_xgb_params}
tuned_xgb = XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
print(f"Tuned XGBoost CV RMSE: {{np.sqrt(-cross_val_score(tuned_xgb, X_train_final, y_train, scoring='neg_mean_squared_error', cv=kf)).mean():.4f}}")
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["## Optimisation Step 3: Tune XGBoost\n", "Tune XGBoost using Optuna for 50 trials."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_step3.split("\n")]})

# Step 4: Stacking
stack_estimators = [
    ('ridge', ridge),
    ('lasso', lasso),
    ('xgb', tuned_xgb)
]
if include_lgbm:
    stack_estimators.append(('lgbm', lgbm))

stacked_model = StackingRegressor(
    estimators=stack_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=kf,
    passthrough=False
)
stacked_score = cv_rmse(stacked_model, X_train_final, y_train)
print(f"| StackingRegressor CV RMSE | {stacked_score:.4f} |")

code_step4 = f"""# Step 4: Implement proper StackingRegressor
from sklearn.ensemble import StackingRegressor

stack_estimators = [
    ('ridge', ridge),
    ('lasso', lasso),
    ('xgb', tuned_xgb)
]
if {include_lgbm}:
    stack_estimators.append(('lgbm', lgbm))

stacked_model = StackingRegressor(
    estimators=stack_estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=kf,
    passthrough=False
)

stacked_score = np.sqrt(-cross_val_score(stacked_model, X_train_final, y_train, scoring="neg_mean_squared_error", cv=kf)).mean()
print(f"StackingRegressor CV RMSE: {{stacked_score:.4f}}")
"""
cells_to_append.append({"cell_type": "markdown", "metadata": {}, "source": ["## Optimisation Step 4: Stacking\n", "Implement a proper StackingRegressor with Ridge as the meta-learner."]})
cells_to_append.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line+"\n" for line in code_step4.split("\n")]})

# Final Selection
best_overall_score = min(opt_blend_tuned_score, stacked_score)
print(f"| BEST OVERALL SCORE | {best_overall_score:.4f} |")

if best_overall_score == stacked_score:
    best_model = stacked_model
else:
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
    best_model = WeightedAveragingModels(models=(ridge, lasso, tuned_xgb, lgbm), weights=res_tuned.x)

best_model.fit(X_train_final, y_train)
final_preds = np.expm1(best_model.predict(X_test_final))
submission = pd.DataFrame({'Id': test_id, 'SalePrice': final_preds})
submission.to_csv('submission.csv', index=False)

# Append to solution.ipynb
with open('solution.ipynb', 'r') as f:
    nb = json.load(f)

nb['cells'].extend(cells_to_append)
with open('solution.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Optimisation Complete and notebook updated!")
