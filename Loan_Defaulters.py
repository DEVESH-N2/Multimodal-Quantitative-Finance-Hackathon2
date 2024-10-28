# 88.59

# Step 1: Import the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Step 2: Load the datasets
train_1 = pd.read_csv('/kaggle/input/piramal2/train_1.csv', low_memory=False)
train_2_1 = pd.read_csv('/kaggle/input/piramal2/train_2_1.csv', low_memory=False)
test_1 = pd.read_csv('/kaggle/input/piramal2/test_1.csv', low_memory=False)
test_2_1 = pd.read_csv('/kaggle/input/piramal2/test_2_1.csv', low_memory=False)

# Ensure 'id' columns are consistent in format
train_1['id'] = train_1['id'].astype(str)
train_2_1['id'] = train_2_1['id'].astype(str)
test_1['id'] = test_1['id'].astype(str)
test_2_1['id'] = test_2_1['id'].astype(str)

# Step 3: Drop non-feature columns
test_loan_ids = test_1['id']
test_ids = test_1['loan_id']
train_1 = train_1.drop(columns=['loan_id'])
test_1 = test_1.drop(columns=['loan_id'])

# Step 4: Apply Ordinal Encoding
categorical_cols_train = train_1.select_dtypes(include=['object']).columns.tolist()
categorical_cols_test = test_1.select_dtypes(include=['object']).columns.tolist()

if 'id' in categorical_cols_train:
    categorical_cols_train.remove('id')
if 'id' in categorical_cols_test:
    categorical_cols_test.remove('id')

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_1[categorical_cols_train] = ordinal_encoder.fit_transform(train_1[categorical_cols_train])
test_1[categorical_cols_test] = ordinal_encoder.transform(test_1[categorical_cols_test])

# Handle missing values
train_1.fillna(0, inplace=True)
test_1.fillna(0, inplace=True)

# Step 7: Drop duplicates in test_2_1 based on 'id'
test_2_1 = test_2_1.drop_duplicates(subset=['id'])
train_2_1 = train_2_1.drop_duplicates(subset=['id'])

# Merge numeric columns
numeric_columns_train_2_1 = train_2_1.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_columns_test_2_1 = test_2_1.select_dtypes(include=['int64', 'float64']).columns.tolist()

train_merged = train_1.merge(train_2_1[numeric_columns_train_2_1 + ['id']], on='id', how='left')
test_merged = test_1.merge(test_2_1[numeric_columns_test_2_1 + ['id']], on='id', how='left')

# Handle missing values in merged datasets
train_merged.fillna(0, inplace=True)
test_merged.fillna(0, inplace=True)

# Step 5: Select numeric columns and prepare data for training
numeric_columns = train_merged.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'label' in numeric_columns:
    numeric_columns.remove('label')

X = train_merged[numeric_columns]
y = train_merged['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train an XGBoost model to get feature importance
xgb_initial_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)
xgb_initial_model.fit(X_train, y_train)

# Get feature importance and select the top 150 features
feature_importances = xgb_initial_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

# Select the top 150 features
top_150_features = feature_importance_df['feature'].head(150).tolist()

# Use only the top 150 features for further model training
X_train_top150 = X_train[top_150_features]
X_val_top150 = X_val[top_150_features]
test_merged_top150 = test_merged[top_150_features]

# Step 7: Initialize CatBoost, XGBoost, and LightGBM classifiers
catboost_model = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', random_seed=42, verbose=200)
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metric='auc', random_state=42)

# Define parameter grids for CatBoost, XGBoost, and LightGBM
param_grid_catboost = {
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.1],
    'iterations': [500, 1000],
    'l2_leaf_reg': [3, 5, 7],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0.2, 0.5, 1.0],
    'random_strength': [1, 1.5, 2],
}

param_grid_xgb = {
    'n_estimators': [600, 800, 1000],
    'learning_rate': [0.005, 0.01, 0.03],
    'max_depth': [8, 10, 12],
    'subsample': [0.8, 0.85, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [3, 5, 7],
    'reg_alpha': [0.01, 0.05, 0.1],
    'reg_lambda': [1.5, 2.5, 3],
    'scale_pos_weight': [1, 2, 3],
    'max_delta_step': [0, 0.5, 1]
}

param_grid_lgb = {
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.03, 0.1],
    'n_estimators': [500, 1000],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8]
}

# Step 8: Perform RandomizedSearchCV for CatBoost, XGBoost, and LightGBM
catboost_random_search = RandomizedSearchCV(estimator=catboost_model, param_distributions=param_grid_catboost, 
                                            scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, random_state=42)
catboost_random_search.fit(X_train_top150, y_train)

xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid_xgb, 
                                       scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, random_state=42)
xgb_random_search.fit(X_train_top150, y_train)

lgb_random_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_grid_lgb, 
                                       scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, random_state=42)
lgb_random_search.fit(X_train_top150, y_train)

# Get the best models
best_catboost_model = catboost_random_search.best_estimator_
best_xgb_model = xgb_random_search.best_estimator_
best_lgb_model = lgb_random_search.best_estimator_

# Step 9: Stacking classifier with CatBoost, XGBoost, and LightGBM
stacking_model = StackingClassifier(estimators=[
    ('catboost', best_catboost_model),
    ('xgboost', best_xgb_model),
    ('lightgbm', best_lgb_model)
], final_estimator=LogisticRegression(), cv=3)

# Train the stacked model
stacking_model.fit(X_train_top150, y_train)

# Step 10: Predictions and evaluation
y_pred_val = stacking_model.predict_proba(X_val_top150)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_val)
print(f"Stacked Model ROC AUC Score (Validation): {roc_auc}")

# Step 11: Make predictions on test data
test_pred = stacking_model.predict_proba(test_merged_top150)[:, 1]

# Ensure the length of predictions and loan_id are the same
if len(test_pred) == len(test_ids):
    submission_df = pd.DataFrame({'loan_id': test_ids, 'loan_default_prob': test_pred})
else:
    print("Error: Mismatch in test data length!")

# Save submission file
submission_df.to_csv('lasthope.csv', index=False)
