import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
import joblib

# ---------------------- LOAD AND BASE FEATURES ----------------------
df = pd.read_csv("data.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
required = {"date","aod","d2m","t2m","u10","v10","sp","tp","PM25"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Time-based
df["month"] = df["date"].dt.month
df["dayofyear"] = df["date"].dt.dayofyear
df["weekday"] = df["date"].dt.weekday

# Wind magnitude/direction & cyclic encoding
df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
df["wind_dir_deg"] = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) + 360) % 360
df["wind_dir_rad"] = np.deg2rad(df["wind_dir_deg"])
df["wind_dir_sin"] = np.sin(df["wind_dir_rad"])
df["wind_dir_cos"] = np.cos(df["wind_dir_rad"])

# Cyclical encodings (seasonality)
df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
df["doy_sin"]   = np.sin(2*np.pi*df["dayofyear"]/366.0)
df["doy_cos"]   = np.cos(2*np.pi*df["dayofyear"]/366.0)
df["weekday_sin"] = np.sin(2*np.pi*df["weekday"]/7.0)
df["weekday_cos"] = np.cos(2*np.pi*df["weekday"]/7.0)

# -------- LAGGED & ROLLING FEATURES FOR D+1 STRUCTURE ----------
def add_lag_feats(data: pd.DataFrame, target="PM25"):
    d = data.copy()
    for L in [1,2,3,7,14]:
        d[f"{target}_lag{L}"] = d[target].shift(L)
    d[f"{target}_roll3_mean"] = d[target].rolling(3).mean()
    d[f"{target}_roll7_mean"] = d[target].rolling(7).mean()
    d[f"{target}_roll7_std"]  = d[target].rolling(7).std()
    return d

df = add_lag_feats(df, target="PM25")
num_features = [
    "aod","d2m","t2m","u10","v10","sp","tp","wind_speed",
    "wind_dir_sin","wind_dir_cos",
    "month_sin","month_cos","doy_sin","doy_cos",
    "weekday_sin","weekday_cos"
]
lag_features = [
    "PM25_lag1","PM25_lag2","PM25_lag3","PM25_lag7","PM25_lag14",
    "PM25_roll3_mean","PM25_roll7_mean","PM25_roll7_std"
]
features = num_features + lag_features
target = "PM25"

df = df.dropna(subset=features + [target]).reset_index(drop=True)
X = df[features]
y = df[target]

# -------------------- TIME-AWARE SPLIT LAST 20% --------------------
cut = int(len(df)*0.8)
X_train, y_train = X.iloc[:cut], y.iloc[:cut]
X_test,  y_test  = X.iloc[cut:],  y.iloc[cut:]

# ---------------------- BASELINE RF & TSCV ---------------------
tscv = TimeSeriesSplit(n_splits=5)
rmse_scorer = make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)))

rf_baseline = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf_baseline, X_train, y_train, cv=tscv, scoring=rmse_scorer, n_jobs=-1)
print("TimeSeries CV RMSE (neg): mean/std:", float(cv_scores.mean()), float(cv_scores.std()))

rf_baseline.fit(X_train, y_train)
y_pred_base = rf_baseline.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base  = mean_absolute_error(y_test, y_pred_base)
r2_base   = r2_score(y_test, y_pred_base)
print("Baseline holdout metrics:", {"RMSE": float(rmse_base), "MAE": float(mae_base), "R2": float(r2_base)})

# -------------------- CONSERVATIVE RF TUNING --------------------
param_grid = {
    "n_estimators": [300, 400, 600],
    "max_depth": [8, 10, 12, None],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [2, 4, 8],
    "max_features": ["sqrt", 0.5]
}
rf_for_search = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
    rf_for_search,
    param_distributions=param_grid,
    n_iter=20,
    cv=tscv,
    scoring=rmse_scorer,
    n_jobs=-1,
    random_state=42,
    refit=True
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("Best params:", search.best_params_)

y_pred_tuned = best_rf.predict(X_test)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned  = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned   = r2_score(y_test, y_pred_tuned)
print("Tuned holdout metrics:", {"RMSE": float(rmse_tuned), "MAE": float(mae_tuned), "R2": float(r2_tuned)})

# -------------------- CHOSEN MODEL PICK --------------------
chosen_model = best_rf if rmse_tuned <= rmse_base else rf_baseline
chosen_label = "tuned" if chosen_model is best_rf else "baseline"
print("Chosen model:", chosen_label)

# ---------------------- PRED INTERVAL (ASCII) ----------------------
def bootstrap_predict(model, X_tr, y_tr, X_te, base_params, n=100, seed=100):
    preds = []
    rng = np.random.RandomState(seed)
    for i in range(n):
        Xb, yb = resample(X_tr, y_tr, random_state=rng.randint(0, 1_000_000))
        m = RandomForestRegressor(**base_params)
        m.set_params(random_state=rng.randint(0, 1_000_000))
        m.fit(Xb, yb)
        preds.append(m.predict(X_te))
    P = np.vstack(preds)
    p5, p50, p95 = np.percentile(P, [5, 50, 95], axis=0)
    return P.mean(axis=0), p5, p50, p95

base_params = {k: v for k,v in chosen_model.get_params().items() if k in RandomForestRegressor().get_params()}
boot_mean, p5, p50, p95 = bootstrap_predict(chosen_model, X_train, y_train, X_test, base_params, n=100)

print("Example prediction interval for first 5 points (ASCII-only):")
for i in range(min(5, len(y_test))):
    dt = df["date"].iloc[cut + i].date()
    print(f"t={dt} truth={y_test.iloc[i]:.2f} pred~{boot_mean[i]:.2f} PI[5%,95%]=[{p5[i]:.2f},{p95[i]:.2f}]")
within = ((y_test.values >= p5) & (y_test.values <= p95)).mean()
print(f"Empirical 90% PI coverage: {within:.3f}")

# ---------------------- SAVE ARTIFACT ----------------------
package = {
    "model": chosen_model,
    "chosen": chosen_label,
    "features": features,
    "notes": {
        "split": "chronological last 20% test",
        "tscv": 5,
        "lags": "1,2,3,7,14",
        "rollings": "3,7 mean; 7 std",
        "wind_dir": "sin/cos",
        "time": "month/day-of-year and weekday cyclical",
        "scorer": "negative RMSE"
    },
    "metrics": {
        "cv_rmse_neg_mean": float(cv_scores.mean()),
        "cv_rmse_neg_std": float(cv_scores.std()),
        "baseline": {"RMSE": float(rmse_base), "MAE": float(mae_base), "R2": float(r2_base)},
        "tuned": {"RMSE": float(rmse_tuned), "MAE": float(mae_tuned), "R2": float(r2_tuned)}
    },
    "best_params": search.best_params_
}
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(package, "artifacts/pm25_d1_rf_package.joblib")
print("Saved to artifacts/pm25_d1_rf_package.joblib")
