# ==================================================
# HOUSE PRICE PREDICTION - FINAL PROFESSIONAL VERSION
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

import joblib

# ==================================================
# STEP 1: Load datasets
# ==================================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

TARGET = 'TARGET(PRICE_IN_LACS)'

# ==================================================
# STEP 2: Separate features & target
# ==================================================
X = train_df.drop([TARGET, "ADDRESS"], axis=1)
y = train_df[TARGET]

test_df = test_df.drop("ADDRESS", axis=1)

# ==================================================
# FEATURE ENGINEERING 1: BHK DENSITY (SAFE)
# ==================================================
X["BHK_DENSITY"] = X["BHK_NO."] / X["SQUARE_FT"]
test_df["BHK_DENSITY"] = test_df["BHK_NO."] / test_df["SQUARE_FT"]

X["BHK_DENSITY"] = X["BHK_DENSITY"].replace([np.inf, -np.inf], np.nan)
test_df["BHK_DENSITY"] = test_df["BHK_DENSITY"].replace([np.inf, -np.inf], np.nan)

# ==================================================
# FEATURE ENGINEERING 2: LOCATION CLUSTERING
# ==================================================
coords = pd.concat([
    X[["LATITUDE", "LONGITUDE"]],
    test_df[["LATITUDE", "LONGITUDE"]]
], axis=0)

kmeans = KMeans(n_clusters=10, random_state=42)
coords["LOCATION_CLUSTER"] = kmeans.fit_predict(coords)

X["LOCATION_CLUSTER"] = coords.iloc[:len(X)]["LOCATION_CLUSTER"].values
test_df["LOCATION_CLUSTER"] = coords.iloc[len(X):]["LOCATION_CLUSTER"].values

# ==================================================
# STEP 3: Combine train & test for encoding
# ==================================================
full_df = pd.concat([X, test_df], axis=0)

cat_cols = full_df.select_dtypes(include="object").columns

full_df_encoded = pd.get_dummies(
    full_df,
    columns=cat_cols,
    drop_first=True
)

X_encoded = full_df_encoded.iloc[:len(X)]
test_encoded = full_df_encoded.iloc[len(X):]

X_encoded = X_encoded.fillna(X_encoded.mean())
test_encoded = test_encoded.fillna(test_encoded.mean())

# ==================================================
# STEP 4: Train-validation split
# ==================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# STEP 5: BASELINE RANDOM FOREST
# ==================================================
base_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

base_model.fit(X_train, y_train)

# ==================================================
# STEP 6: HOLD-OUT VALIDATION
# ==================================================
val_preds = base_model.predict(X_val)

print("Hold-out MAE:", mean_absolute_error(y_val, val_preds))
print("Hold-out RMSE:", np.sqrt(mean_squared_error(y_val, val_preds)))
print("Hold-out R2:", r2_score(y_val, val_preds))

# ==================================================
# PLOT 1: Actual vs Predicted
# ==================================================
plt.figure(figsize=(6, 6))
plt.scatter(y_val, val_preds, alpha=0.4)
plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()],
         color="red")
plt.xlabel("Actual Price (LACS)")
plt.ylabel("Predicted Price (LACS)")
plt.title("Actual vs Predicted Prices")
plt.show()

# ==================================================
# PLOT 2: Error Distribution
# ==================================================
errors = y_val - val_preds

plt.figure(figsize=(6, 4))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error (LACS)")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.show()

# ==================================================
# STEP 7: CROSS-VALIDATION
# ==================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(
    base_model,
    X_encoded,
    y,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

cv_mae = -cross_val_score(
    base_model,
    X_encoded,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("CV R2 Scores:", cv_r2)
print("Mean CV R2:", cv_r2.mean())
print("CV MAE Scores:", cv_mae)
print("Mean CV MAE:", cv_mae.mean())

# ==================================================
# STEP 8: HYPERPARAMETER TUNING (GRID SEARCH)
# ==================================================
param_grid = {
    "n_estimators": [150, 250],
    "max_depth": [None, 15, 25],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_encoded, y)

print("Best Parameters:", grid_search.best_params_)
print("Best CV R2:", grid_search.best_score_)

# ==================================================
# STEP 9: FINAL OPTIMIZED MODEL
# ==================================================
best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)

opt_preds = best_model.predict(X_val)

print("Optimized MAE:", mean_absolute_error(y_val, opt_preds))
print("Optimized RMSE:", np.sqrt(mean_squared_error(y_val, opt_preds)))
print("Optimized R2:", r2_score(y_val, opt_preds))

# ==================================================
# STEP 10: TEST SET PREDICTION
# ==================================================
test_predictions = best_model.predict(test_encoded)

submission = pd.DataFrame({
    "Id": test_df.index,
    "Predicted_Price_LACS": test_predictions
})

submission.to_csv("predictions_rf_optimized.csv", index=False)

# ==================================================
# STEP 11: FEATURE IMPORTANCE
# ==================================================
importance_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(importance_df.head(10))

importance_df.head(10).plot(
    kind="barh",
    x="Feature",
    y="Importance",
    figsize=(8, 5),
    legend=False
)
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.show()

# ==================================================
# STEP 12: SAVE FINAL MODEL
# ==================================================
joblib.dump(best_model, "house_price_rf_optimized.pkl")
print("Final optimized model saved successfully")

import joblib
joblib.dump(X_encoded.columns, "columns.pkl")
