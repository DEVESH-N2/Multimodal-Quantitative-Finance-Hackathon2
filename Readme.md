Introduction

This project aims to build a predictive model for loan default detection. Multiple datasets were processed, and an ensemble approach using CatBoost, XGBoost, and LightGBM classifiers was employed. The final model was built using a stacking classifier, combining the strengths of these classifiers for improved performance. This report outlines the approach taken, the preprocessing steps, and the models used in this project.

Data Preprocessing

Data Loading:
Four CSV files (train_1.csv, train_2_1.csv, test_1.csv, test_2_1.csv) were loaded, containing both train and test data.
Datasets were cleaned by ensuring consistency in the 'id' columns, handling missing values, and merging based on the 'id'.

Feature Engineering:
Ordinal Encoding: Categorical columns were identified and transformed into numerical representations using OrdinalEncoder. The encoded columns replaced the original categorical columns in both training and test datasets.

Handling Missing Values: Missing values were filled with zeros in both train and test datasets.

Merging Datasets: Numeric features from train_2_1 and test_2_1 were merged with train_1 and test_1 based on the 'id' column.
Feature Selection:

An initial XGBoost model was trained to assess feature importance. The top 150 features were selected based on their importance scores, which were used for further model training.

Models and Techniques

CatBoost Classifier:
CatBoost is known for its efficient handling of categorical data and its superior performance on structured data. A parameter grid was defined, and RandomizedSearchCV was used to optimize hyperparameters like depth, learning_rate, and iterations.

XGBoost Classifier:
XGBoost is a powerful gradient boosting algorithm. Similar to CatBoost, a hyperparameter tuning process was conducted using RandomizedSearchCV. Key parameters such as n_estimators, max_depth, and subsample were tuned.

LightGBM Classifier:
LightGBM is another gradient boosting method optimized for speed and efficiency. A grid search was performed to tune parameters like num_leaves, max_depth, and learning_rate.

Stacking Classifier:
After training and tuning each individual model, a stacking classifier was implemented to combine the predictions of CatBoost, XGBoost, and LightGBM. A Logistic Regression model was used as the final estimator to make the overall prediction based on the outputs of the base models.

Model Evaluation
Validation: The dataset was split into training and validation sets (80-20 split). The stacked model was evaluated using the ROC AUC Score, achieving strong performance on the validation data.
Validation ROC AUC Score: 89.05

Prediction on Test Data
After training the stacked model, predictions were made on the test data (test_merged_top150), and the predicted probabilities of loan default were generated.
Test ROC AUC Score: 88.59

A submission file was created in the required format, containing the loan_id and the corresponding default probabilities.

Conclusion
The ensemble approach using CatBoost, XGBoost, and LightGBM, combined through a stacking classifier, allowed for robust and accurate predictions of loan defaults. The model was optimized using extensive hyperparameter tuning, feature selection, and advanced machine learning techniques.