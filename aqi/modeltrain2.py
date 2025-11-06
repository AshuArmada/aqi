import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset (features must include those listed below)
df = pd.read_csv("data.csv", parse_dates=['date'])

# Feature engineering used for modeling
num_features = ['lat','lon','aod','d2m','t2m','u10','v10','sp','tp']
df['month'] = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear
df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
df['wind_dir_deg'] = (np.degrees(np.arctan2(-df['u10'], -df['v10'])) + 360) % 360

features = num_features + ['month','dayofyear','wind_speed','wind_dir_deg']
target = 'PM25'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Random Forest training
rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Baseline evaluation on holdout
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Baseline holdout metrics:", {"RMSE": rmse, "MAE": mae, "R2": r2})

# Cross-validation (5-fold) on full dataset
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2", n_jobs=-1)
print("CV R2 mean and std:", cv_scores.mean(), cv_scores.std())

# Feature importances
feat_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("Top feature importances:")
print(feat_importance.head(10).to_string())

# Hyperparameter tuning (RandomizedSearchCV)
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
search = RandomizedSearchCV(
    rf, param_distributions=param_grid, n_iter=10, cv=3,
    scoring="r2", n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("Best params:", search.best_params_)

# Evaluate tuned model on holdout
y_pred_best = best_rf.predict(X_test)
mse_b = mean_squared_error(y_test, y_pred_best)
rmse_b = np.sqrt(mse_b)
mae_b = mean_absolute_error(y_test, y_pred_best)
r2_b = r2_score(y_test, y_pred_best)
print("Tuned holdout metrics:", {"RMSE": rmse_b, "MAE": mae_b, "R2": r2_b})

# Save models and metadata
model_package = {
    "baseline_model": rf,
    "tuned_model": best_rf,
    "features": features,
    "metrics": {
        "baseline": {"RMSE": rmse, "MAE": mae, "R2": r2},
        "tuned": {"RMSE": rmse_b, "MAE": mae_b, "R2": r2_b}
    },
    "search_best_params": search.best_params_
}
joblib.dump(model_package, "pm25_rf_package.joblib")
print("Saved model package: pm25_rf_package.joblib")
