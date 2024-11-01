# 89.04

# Step 1: Import the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Step 2: Load the datasets
train_1 = pd.read_csv('/kaggle/input/piramal3/train_1.csv', low_memory=False)
train_2_1 = pd.read_csv('/kaggle/input/piramal3/train_2_1.csv', low_memory=False)
test_1 = pd.read_csv('/kaggle/input/piramal3/test_1.csv', low_memory=False)
test_2_1 = pd.read_csv('/kaggle/input/piramal3/test_2_1.csv', low_memory=False)

# Step 3: Ensure 'id' columns are consistent in format
train_1['id'] = train_1['id'].astype(str)
train_2_1['id'] = train_2_1['id'].astype(str)
test_1['id'] = test_1['id'].astype(str)
test_2_1['id'] = test_2_1['id'].astype(str)

# Step 4: Drop non-feature columns but keep 'loan_id' in test for final output
test_loan_ids = test_1['id']
test_ids = test_1['loan_id']
train_1 = train_1.drop(columns=['loan_id'])
test_1 = test_1.drop(columns=['loan_id'])

# Step 5: Ordinal Encoding for categorical features in train_1 and test_1
categorical_cols_train = train_1.select_dtypes(include=['object']).columns.tolist()
categorical_cols_test = test_1.select_dtypes(include=['object']).columns.tolist()

# Remove 'id' column from the list of categorical columns
if 'id' in categorical_cols_train:
    categorical_cols_train.remove('id')
if 'id' in categorical_cols_test:
    categorical_cols_test.remove('id')

# Apply OrdinalEncoder to categorical columns
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_1[categorical_cols_train] = ordinal_encoder.fit_transform(train_1[categorical_cols_train])
test_1[categorical_cols_test] = ordinal_encoder.transform(test_1[categorical_cols_test])

# Step 6: Handle missing values (set to 0)
train_1.fillna(0, inplace=True)
test_1.fillna(0, inplace=True)

# Step 7: Drop duplicates in test_2_1 based on 'id'
test_2_1 = test_2_1.drop_duplicates(subset=['id'])

# Step 8: Merge only int and float columns from train_2_1 and test_2_1
numeric_columns_train_2_1 = train_2_1.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_columns_test_2_1 = test_2_1.select_dtypes(include=['int64', 'float64']).columns.tolist()

train_merged = train_1.merge(train_2_1[numeric_columns_train_2_1 + ['id']], on='id', how='left')
test_merged = test_1.merge(test_2_1[numeric_columns_test_2_1 + ['id']], on='id', how='left')

# Step 9: Handle missing values in merged datasets (set to 0)
train_merged.fillna(0, inplace=True)
test_merged.fillna(0, inplace=True)

# Step 10: Select only integer and float columns for the model
numeric_columns = train_merged.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'label' in numeric_columns:
    numeric_columns.remove('label')

# Step 11: Prepare the feature matrix (X) and target vector (y)
X = train_merged[numeric_columns]
y = train_merged['label']

# Step 12: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 13: Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)

param_dist = {
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

# Step 15: Perform randomized search with cross-validation
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, 
                                   scoring='roc_auc', cv=3, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Get the best model from randomized search
best_xgb_model = random_search.best_estimator_

# Step 16: Train the best model and calculate feature importance
best_xgb_model.fit(X_train, y_train)

# Get feature importance scores
feature_importances = best_xgb_model.feature_importances_

# Create a DataFrame to store features and their importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort features by importance (descending order)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Step 17: Select the top 200 features
top_200_features = feature_importance_df.head(150)['Feature'].tolist()

# Step 18: Create new statistical features using top 200 features
# Create a new DataFrame for the new features
X_train_top_200 = X_train[top_200_features]
X_val_top_200 = X_val[top_200_features]
X_test_top_200 = test_merged[top_200_features]

# Generate new statistical features (mean, std, etc.) and prefix 'c_' to new feature names
for func in ['mean', 'std', 'min', 'max', 'median']:
    X_train_top_200[f'c_{func}'] = X_train_top_200.apply(eval(f'np.{func}'), axis=1)
    X_val_top_200[f'c_{func}'] = X_val_top_200.apply(eval(f'np.{func}'), axis=1)
    X_test_top_200[f'c_{func}'] = X_test_top_200.apply(eval(f'np.{func}'), axis=1)

# Step 19: Retrain the model using top 200 features + new statistical features
best_xgb_model.fit(X_train_top_200, y_train, eval_set=[(X_val_top_200, y_val)], 
                   early_stopping_rounds=10, verbose=True)

# Step 20: Make predictions on the validation set
y_pred_val = best_xgb_model.predict_proba(X_val_top_200)[:, 1]

# Calculate ROC AUC score on validation data
roc_auc = roc_auc_score(y_val, y_pred_val)
print(f"XGBoost ROC AUC Score (Validation) with Top 200 Features + Statistics: {roc_auc}")

# Step 21: Make predictions on the test data with top 200 features + statistics
test_pred = best_xgb_model.predict_proba(X_test_top_200)[:, 1]

# Ensure the length of predictions and loan_id are the same
if len(test_pred) != len(test_loan_ids):
    raise ValueError(f"Prediction length {len(test_pred)} doesn't match loan_id length {len(test_loan_ids)}")

# Step 22: Create output DataFrame and save predictions to CSV
output_df = pd.DataFrame({
    'loan_id': test_ids,
    'prob': test_pred
})
output_df.to_csv('holy.csv', index=False)
print("Predictions saved to predictions_top_200_with_stats.csv")

# Step 23: Plot feature importance of top features
xgb.plot_importance(best_xgb_model, importance_type='weight', max_num_features=10, height=0.8)
plt.title('Top 10 Feature Importances')
plt.show()
