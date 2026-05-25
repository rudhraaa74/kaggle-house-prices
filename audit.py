import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

audit_results = {}

# 1.1 Target Variable
sale_price = train['SalePrice']
audit_results['target'] = {
    'skew': float(sale_price.skew()),
    'range': [float(sale_price.min()), float(sale_price.max())],
    'mean': float(sale_price.mean()),
    'median': float(sale_price.median())
}

# 1.2 Missing Values
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0)
missing = (all_data.isnull().sum() / len(all_data)) * 100
missing = missing[missing > 0].sort_values(ascending=False)
audit_results['missing'] = missing.to_dict()

# 1.4 Outliers (Focusing on GrLivArea which is famous in this dataset, plus finding others)
num_cols = train.select_dtypes(include=[np.number]).columns
corrs = train[num_cols].corr()['SalePrice'].sort_values(ascending=False)
audit_results['correlations'] = corrs.to_dict()

# Checking for highly correlated features (multicollinearity)
corr_matrix = train[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        if upper.loc[row, col] > 0.8:
            high_corr_pairs.append((row, col, float(upper.loc[row, col])))
audit_results['multicollinearity'] = high_corr_pairs

# Outliers check - print max z-scores for top correlated features
top_features = corrs.index[1:6] # excluding SalePrice
outliers = {}
for feat in top_features:
    z = np.abs(stats.zscore(train[feat].dropna()))
    outliers[feat] = int((z > 4).sum())
audit_results['outliers_z4'] = outliers

# Specifically check GrLivArea > 4000 vs SalePrice < 300000
weird_houses = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].shape[0]
audit_results['grlivarea_outliers'] = weird_houses

# Print as JSON
print(json.dumps(audit_results, indent=2))
