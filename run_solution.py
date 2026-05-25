import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop extreme outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# Drop zero-variance column
train.drop(['Utilities'], axis=1, inplace=True)
test.drop(['Utilities'], axis=1, inplace=True)

y_train = np.log1p(train['SalePrice'])
train_features = train.drop(['Id', 'SalePrice'], axis=1)
test_features = test.drop(['Id'], axis=1)
test_id = test['Id']

print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(train['SalePrice'], kde=True)
plt.title('SalePrice Distribution')
plt.subplot(1,2,2)
sns.histplot(y_train, kde=True)
plt.title('Log(SalePrice) Distribution')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])
plt.title('GrLivArea vs SalePrice (Anomalies removed)')
plt.show()

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.frontage_medians_ = X.groupby('Neighborhood')['LotFrontage'].median()
        self.overall_frontage_median_ = X['LotFrontage'].median()
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        # 1. Impute LotFrontage properly based on neighborhood
        X_out['LotFrontage'] = X_out.apply(
            lambda row: self.frontage_medians_.get(row['Neighborhood'], self.overall_frontage_median_) 
            if pd.isnull(row['LotFrontage']) else row['LotFrontage'], axis=1
        )
        
        # 2. Feature Engineering
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
genuine_cat = ['MSZoning', 'Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 
               'Functional', 'SaleType']

class NA_Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.modes_ = X[genuine_cat].mode().iloc[0]
        return self
        
    def transform(self, X):
        X_out = X.copy()
        # Semantic fills
        X_out[semantic_cat] = X_out[semantic_cat].fillna("None")
        X_out[semantic_num] = X_out[semantic_num].fillna(0)
        # Genuine missing categorical mode fills
        for col in genuine_cat:
            X_out[col] = X_out[col].fillna(self.modes_[col])
            
        # Fallback for remaining features
        str_cols = X_out.select_dtypes(include=['object', 'string', 'category']).columns
        X_out[str_cols] = X_out[str_cols].fillna("None")
        num_cols = X_out.select_dtypes(exclude=['object', 'string', 'category']).columns
        X_out[num_cols] = X_out[num_cols].fillna(0)
        return X_out

# Preserve ordinal order instead of one-hot encoding
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

# Final scaling (RobustScaler handles remaining outliers) and One-Hot Encoding
final_transformer = ColumnTransformer(transformers=[
    ('num', RobustScaler(), num_cols),
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nom_cols)
])

X_train_final = final_transformer.fit_transform(X_train_inter)
X_test_final = final_transformer.transform(X_test_inter)
print(f"Final train shape after OHE: {X_train_final.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_rmse(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# Ridge - tuned alpha to 10 for stability against multicollinearity
ridge = Ridge(alpha=10)
# Lasso - tuned alpha to 0.0005 to force sparsity without losing too much signal
lasso = Lasso(alpha=0.0005, random_state=42)
# XGBoost - constrained depth and learning rate for generalisation
xgb = XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=3, 
                   subsample=0.8, colsample_bytree=0.8, random_state=42)

models = {'Ridge': ridge, 'Lasso': lasso, 'XGBoost': xgb}
scores = {}
for name, model in models.items():
    score = cv_rmse(model, X_train_final, y_train)
    scores[name] = score.mean()
    print(f"{name}: Mean RMSE = {score.mean():.4f}, Std = {score.std():.4f}")

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

blended_model = AveragingModels(models=(ridge, lasso, xgb))
blend_score = cv_rmse(blended_model, X_train_final, y_train)
print(f"Blended Model CV RMSE: {blend_score.mean():.4f} +/- {blend_score.std():.4f}")

blended_model.fit(X_train_final, y_train)
test_preds = np.expm1(blended_model.predict(X_test_final))

submission = pd.DataFrame({'Id': test_id, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)

print("First 5 rows of submission:")
print(submission.head())

print("========== FINAL SUMMARY ==========")
print("Final model architecture: Simple Averaging Blend of Ridge (alpha=10), Lasso (alpha=0.0005), and XGBoost (depth=3, lr=0.05).")
print(f"CV RMSE score: {blend_score.mean():.4f}")

# Get Lasso coefficients to see key features
lasso.fit(X_train_final, y_train)
feature_names = final_transformer.get_feature_names_out()
coefs = pd.Series(lasso.coef_, index=feature_names)
print("\nKey features that mattered most (Lasso Coefficients):")
print("Top Positive Contributors:")
print(coefs.sort_values(ascending=False).head(5))
print("\nTop Negative Contributors:")
print(coefs.sort_values(ascending=True).head(5))

print("\nCaveats / Next Steps:")
print("- Next steps would involve more extensive hyperparameter tuning using Optuna.")
print("- Adding a meta-model like StackingRegressor instead of a simple average might yield better generalisation.")

