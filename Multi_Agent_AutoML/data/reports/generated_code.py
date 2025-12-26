import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import xgboost as xgb

# Load data
fp = "/Users/nikoloz/Documents/Personal/LLMs/Multi_Agent_AutoML/data/interim/engineered_data.csv"
df = pd.read_csv(fp)

# Target and features
target_col = 'Churn'
X = df.drop(columns=[target_col])
y = df[target_col]

# Determine task type
a_is_classification = y.nunique() < 20

# Split data
if a_is_classification:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

if a_is_classification:
    # Handle imbalance via scale_pos_weight
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg) / float(max(pos, 1))

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.06,
        max_depth=6,
        n_estimators=500,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
else:
    model = xgb.XGBRegressor(
        learning_rate=0.06,
        max_depth=6,
        n_estimators=500,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.4f}")
