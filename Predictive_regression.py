import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Bike Sharing Demand Analysis - Question 1")
print("="*60)

# Loading the data
df = pd.read_csv("train.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# Adding time-based features
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year

# Removing columns (casual and registered are target leakage)
drop_cols = ["datetime", "casual", "registered"]
X = df.drop(drop_cols + ["count"], axis=1)
y = df["count"].values

print(f"\nUsing features: {list(X.columns)}")
print(f"Total data points: {len(X)}")

# Train/test split: first 80% for training, last 20% for testing
split_point = int(0.8 * len(df))
X_train = X.iloc[:split_point].copy()
X_test = X.iloc[split_point:].copy()
y_train = y[:split_point]
y_test = y[split_point:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Standardize features (fit scaler on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper function to evaluate model
def evaluate_model(model_name, predictions, actual):
    predictions = np.maximum(predictions, 0)  # bike counts can't be negative
    mse_val = mean_squared_error(actual, predictions)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(actual, predictions)
    print(f"{model_name:40} RMSE: {rmse_val:7.2f}  R²: {r2_val:.4f}")
    return {"model": model_name, "mse": mse_val, "rmse": rmse_val, "r2": r2_val}

# Function to create polynomial features WITHOUT interactions
def create_polynomial_no_interactions(X, degree):
    """
    Creates polynomial features up to 'degree' for each variable independently.
    For n features and degree d: returns [x1, x1^2, ..., x1^d, x2, x2^2, ..., x2^d, ...]
    """
    n_samples, n_features = X.shape
    # For each feature, create powers from 1 to degree
    poly_features = []
    for d in range(1, degree + 1):
        poly_features.append(X ** d)
    return np.hstack(poly_features)

all_results = []

print("\n" + "="*60)
print("Testing different models on the test set:")
print("="*60)

# Model 1: Linear Regression (Baseline)
print("\n[1] Linear Regression (Baseline)")
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
lin_predictions = lin_model.predict(X_test_scaled)
all_results.append(evaluate_model("Linear Regression", lin_predictions, y_test))

# Models 2-4: Polynomial Regression WITHOUT interactions
print("\n[2] Polynomial Models (without interactions)")
for deg in [2, 3, 4]:
    X_train_poly = create_polynomial_no_interactions(X_train_scaled, deg)
    X_test_poly = create_polynomial_no_interactions(X_test_scaled, deg)
    
    print(f"   Degree {deg}: {X_train_poly.shape[1]} features (original: {X_train_scaled.shape[1]})")
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_predictions = poly_model.predict(X_test_poly)
    
    model_desc = f"Polynomial degree {deg} (no interactions)"
    all_results.append(evaluate_model(model_desc, poly_predictions, y_test))

# Model 5: Quadratic WITH interactions
print("\n[3] Quadratic Model (with interactions)")
poly2_interact = PolynomialFeatures(degree=2, include_bias=False)
X_train_quad = poly2_interact.fit_transform(X_train_scaled)
X_test_quad = poly2_interact.transform(X_test_scaled)

print(f"   Quadratic with interactions: {X_train_quad.shape[1]} features")

quad_model = LinearRegression()
quad_model.fit(X_train_quad, y_train)
quad_predictions = quad_model.predict(X_test_quad)
all_results.append(evaluate_model("Quadratic with interactions", quad_predictions, y_test))

# Summary
print("\n" + "="*60)
print("Summary of Results (sorted by R²)")
print("="*60)

results_table = pd.DataFrame(all_results).sort_values("r2", ascending=False)
for idx, row in results_table.iterrows():
    print(f"{row['model']:40} RMSE: {row['rmse']:7.2f}  R²: {row['r2']:.4f}")

# Best model
best_result = results_table.iloc[0]
print(f"\n{'='*60}")
print(f"Best performing model: {best_result['model']}")
print(f"R² score: {best_result['r2']:.4f}")
print(f"RMSE: {best_result['rmse']:.2f}")
print(f"{'='*60}")

# Interpretation
print("\nKey Insights:")
print("- Linear regression provides baseline performance")
print("- Polynomial models (no interactions) capture non-linear single-variable effects")
print("- Quadratic with interactions models feature dependencies (e.g., temp × humidity)")
print("- Balance between model complexity and generalization is crucial")
print("\nBias-Variance Trade-off:")
print("- Low degree → High bias (underfitting), Low variance")
print("- High degree → Low bias, High variance (overfitting risk)")
print("- Interactions → Can improve fit if features genuinely interact")