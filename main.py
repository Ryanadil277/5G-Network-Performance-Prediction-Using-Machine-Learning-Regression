import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
### Loading the data
#read data
df = pd.read_csv("data2/Quality of Service 5G.csv")

print(df.head())
print(df.shape)
### Exploratory Data Analysis (EDA) 
print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())
##target 
if "Latency (ms)" in df.columns:
    target = "Latency (ms)"
elif "Latency" in df.columns:
    target = "Latency"
else:
    raise ValueError("Latency column was not found.")

print("\nTarget:")
print(target)
## X Y
y = pd.to_numeric(
    df[target].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
    errors="coerce"
)

valid_rows = y.notna()

df = df.loc[valid_rows].copy()
y = y.loc[valid_rows]

X = df.drop(columns=[target])

if "Timestamp" in X.columns:
    X = X.drop(columns=["Timestamp"])

print("\nFeatures:")
print(X.columns.tolist())

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)
### Numerical and categorical features
categorical_features = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)
### Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)
###linear regression
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n========== Linear Regression ==========")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)
### Random forest regresstor
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\n========== Random Forest ==========")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R2 Score": [linear_r2, rf_r2]
})

print("\n========== Model Comparison ==========")
print(results)
### actual vs predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    linear_predictions,
    alpha=0.5
)

plt.xlabel("Actual Latency")
plt.ylabel("Predicted Latency")
plt.title("Actual vs Predicted Latency - Linear Regression")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()


 ### model improvement
from sklearn.model_selection import GridSearchCV
param_grid = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10]
}
grid_search = GridSearchCV(
    random_forest_model,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation R2:")
print(grid_search.best_score_)
#### Evaluate the best model on the test set
best_rf = grid_search.best_estimator_

best_rf_predictions = best_rf.predict(X_test)

best_rf_mae = mean_absolute_error(
    y_test,
    best_rf_predictions
)

best_rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        best_rf_predictions
    )
)

best_rf_r2 = r2_score(
    y_test,
    best_rf_predictions
)

print("\n========== Tuned Random Forest ==========")
print("MAE:", best_rf_mae)
print("RMSE:", best_rf_rmse)
print("R2 Score:", best_rf_r2)
#### save the best model
####cross validation for linear regression
from sklearn.model_selection import cross_val_score

linear_cv_scores = cross_val_score(
    linear_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n========== Linear Regression Cross-Validation ==========")
print("R2 Scores:", linear_cv_scores)
print("Mean R2:", linear_cv_scores.mean())
print("Standard Deviation:", linear_cv_scores.std())
### residual analysis for linear regression
residuals = y_test - linear_predictions

print("\n========== Residual Analysis ==========")
print("Mean Residual:", residuals.mean())
print("Maximum Error:", residuals.abs().max())
print("Minimum Error:", residuals.abs().min())
####residual plot
plt.figure(figsize=(8, 6))

plt.scatter(
    linear_predictions,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Latency")
plt.ylabel("Residual Error")
plt.title("Residual Analysis - Linear Regression")

plt.show()
#### BEST MODEL 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.01),
    "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "Huber Regression": HuberRegressor(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}
results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n========== Model Comparison ==========")
print(results_df.to_string(index=False))
#### results visualization
import matplotlib.pyplot as plt

# Model Comparison Graph

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# R2
axes[0].bar(results_df["Model"], results_df["R2"])
axes[0].set_title("R² Score Comparison")
axes[0].set_ylabel("R²")
axes[0].tick_params(axis="x", rotation=70)

# MAE
axes[1].bar(results_df["Model"], results_df["MAE"])
axes[1].set_title("MAE Comparison")
axes[1].set_ylabel("MAE (ms)")
axes[1].tick_params(axis="x", rotation=70)

# RMSE
axes[2].bar(results_df["Model"], results_df["RMSE"])
axes[2].set_title("RMSE Comparison")
axes[2].set_ylabel("RMSE (ms)")
axes[2].tick_params(axis="x", rotation=70)

plt.tight_layout()
plt.show()

#### lasso hyperparameter tuning
# Lasso Hyperparameter Tuning

lasso_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(max_iter=10000))
    ]
)

lasso_params = {
    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
}

lasso_grid = GridSearchCV(
    lasso_pipeline,
    lasso_params,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

lasso_grid.fit(X_train, y_train)

print("\n========== Tuned Lasso ==========")
print("Best Alpha:", lasso_grid.best_params_)
print("Best Cross-Validation R2:", lasso_grid.best_score_)

lasso_final = lasso_grid.best_estimator_

lasso_predictions = lasso_final.predict(X_test)

print("Test R2:", r2_score(y_test, lasso_predictions))
print("Test MAE:", mean_absolute_error(y_test, lasso_predictions))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, lasso_predictions)))
#### final lasso model
# Final Lasso Model

final_model = lasso_grid.best_estimator_

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

print("\n========== FINAL MODEL ==========")
print("Model: Lasso Regression (L1)")
print("Best Alpha:", lasso_grid.best_params_["model__alpha"])

print("R2 Score:", r2_score(y_test, final_predictions))
print("MAE:", mean_absolute_error(y_test, final_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, final_predictions)))
##### actual vs predicted plot for final model
# Actual vs Predicted Latency - Final Lasso Model
# Actual vs Predicted - Final Lasso Model

lasso_final = lasso_grid.best_estimator_

lasso_predictions = lasso_final.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    lasso_predictions,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Latency (ms)")
plt.ylabel("Predicted Latency (ms)")
plt.title("Actual vs Predicted Latency - Lasso Regression")

plt.tight_layout()
plt.show()
