import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the insurance dataset
data = pd.read_csv('../../Desktop/insurance.csv')

# Show basic info and preview of the dataset
print(data.head())
print(data.info(0))
print('Statistics Summary:', data.describe())
print(f"Dataset contains {data.shape[0]} rows and {data.shape[1]}columns")
# Check if there is any missing values
print(pd.DataFrame({'percent_missing': data.isnull().sum() * 100 / len(data)}))



# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)


# Features and target
X = data.drop("charges", axis=1)
y = data["charges"]

# Split the data
X_train_pre, X_test_pre, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_pre)
X_test = scaler.transform(X_test_pre)


# Store results
results = []

# ---- Linear Regression ----
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

results.append({
    "Model": "Linear Regression",
    "R2": r2_score(y_test, y_pred_lr),
    "MSE": mean_squared_error(y_test, y_pred_lr),
    "MAE": mean_absolute_error(y_test, y_pred_lr),
    "Features": X_train.shape[1]
})

# --- Coefficients & Performance ---
print('\n--- Linear Regression Summary ---')
print('Intercept:', round(lr.intercept_, 2))
print('Coefficients:', np.round(lr.coef_, 2))
print('R2 score: %.2f' % r2_score(y_test, y_pred_lr))
print('Mean Absolute Error: %.2f' % mean_absolute_error(y_test, y_pred_lr))
print('Mean Squared Error: %.2f' % mean_squared_error(y_test, y_pred_lr))


# ---- Lasso Regression (experiment with a few alphas) ----
for alpha in [0.1, 1.0, 10.0]:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    results.append({
        "Model": f"Lasso (alpha={alpha})",
        "R2": r2_score(y_test, y_pred_lasso),
        "MSE": mean_squared_error(y_test, y_pred_lasso),
        "MAE": mean_absolute_error(y_test, y_pred_lasso),
        "Features": np.sum(lasso.coef_ != 0)
    })

# Print Lasso summary
    print(f"\n--- Lasso Regression Summary (alpha={alpha}) ---")
    print('Intercept:', round(lasso.intercept_, 2))
    print('Coefficients:', np.round(lasso.coef_, 2))
    print('R2 score: %.2f' % r2_score(y_test, y_pred_lasso))
    print('Mean Absolute Error: %.2f' % mean_absolute_error(y_test, y_pred_lasso))
    print('Mean Squared Error: %.2f' % mean_squared_error(y_test, y_pred_lasso))

# Optional: Show which coefficients were set to zero
zeroed_features = np.array(X.columns)[lasso.coef_ == 0]
print("Zeroed-out features:", zeroed_features)


# ---- Polynomial Regression (degree=2) ----
polyDegree=2
poly2 = PolynomialFeatures(degree=polyDegree)
X_poly2_train = poly2.fit_transform(X_train)
X_poly2_test = poly2.transform(X_test)

# Get feature names immediately after transforming ( for printing the weights)
feature_names_poly2 = poly2.get_feature_names_out(input_features=X.columns)

poly2_lr = LinearRegression()
poly2_lr.fit(X_poly2_train, y_train)
y_pred_poly2 = poly2_lr.predict(X_poly2_test)

results.append({
    "Model": "Polynomial Regression (degree=2)",
    "R2": r2_score(y_test, y_pred_poly2),
    "MSE": mean_squared_error(y_test, y_pred_poly2),
    "MAE": mean_absolute_error(y_test, y_pred_poly2),
    "Features": X_poly2_train.shape[1]
})

# --- Coefficients & Performance for Polynomial Degree 2 ---
print('\n--- Polynomial Regression (Degree=2) Summary ---')
print('Intercept:', round(poly2_lr.intercept_, 2))
print('R2 score: %.2f' % r2_score(y_test, y_pred_poly2))
print('Mean Absolute Error: %.2f' % mean_absolute_error(y_test, y_pred_poly2))
print('Mean Squared Error: %.2f' % mean_squared_error(y_test, y_pred_poly2))


# ---- Polynomial Regression (degree=3) ----
polyDegree=3
poly3 = PolynomialFeatures(degree=polyDegree)
X_poly3_train = poly3.fit_transform(X_train)
X_poly3_test = poly3.transform(X_test)


# Get feature names immediately after transforming (for printing the weights)
feature_names_poly3 = poly3.get_feature_names_out(input_features=X.columns)

poly3_lr = LinearRegression()
poly3_lr.fit(X_poly3_train, y_train)
y_pred_poly3 = poly3_lr.predict(X_poly3_test)

results.append({
    "Model": "Polynomial Regression (degree=3)",
    "R2": r2_score(y_test, y_pred_poly3),
    "MSE": mean_squared_error(y_test, y_pred_poly3),
    "MAE": mean_absolute_error(y_test, y_pred_poly3),
    "Features": X_poly3_train.shape[1]
})

# --- Coefficients & Performance for Polynomial Degree 2 ---
print('\n--- Polynomial Regression (Degree=3) Summary ---')
print('Intercept:', round(poly3_lr.intercept_, 2))
print('R2 score: %.2f' % r2_score(y_test, y_pred_poly3))
print('Mean Absolute Error: %.2f' % mean_absolute_error(y_test, y_pred_poly3))
print('Mean Squared Error: %.2f' % mean_squared_error(y_test, y_pred_poly3))



# --- Final Results Table ---
results_df = pd.DataFrame(results)
print("\n--- Regression Results ---")
print(results_df)
results_df.to_csv("results_df.csv", index=False)



# --- Plotting --- #
import matplotlib.pyplot as plt

# Create subplots for each model's predictions
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()

models_preds = [
    ("Linear", y_pred_lr),
    ("Lasso (α=0.1)", Lasso(alpha=0.1).fit(X_train, y_train).predict(X_test)),
    ("Lasso (α=10.0)", Lasso(alpha=10.0).fit(X_train, y_train).predict(X_test)),
    ("Poly (d=2)", y_pred_poly2),
    ("Poly (d=3)", y_pred_poly3)
]

for i, (label, preds) in enumerate(models_preds):
    axs[i].scatter(y_test, preds, alpha=0.5)
    axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    axs[i].set_title(f"{label}: Actual vs Predicted")
    axs[i].set_xlabel("Actual")
    axs[i].set_ylabel("Predicted")
    axs[i].grid(True)

# Hide unused subplot
fig.delaxes(axs[5])
fig.tight_layout()
plt.show()

################## Extra part to print coefficients/ weights ###################################
print("\n" + "="*60)
print("Top 5 Most Influential Features per Model (by absolute weight)")
print("="*60)

# ---------- Linear Regression ----------
linear_weights = pd.DataFrame({'Feature': X.columns, 'Weight': lr.coef_})
top_linear = linear_weights.reindex(linear_weights.Weight.abs().sort_values(ascending=False).index).head(5)
print("\nLinear Regression:")
print(top_linear.to_string(index=False))

# ---------- Lasso Regressions (α = 0.1, 1.0, 10.0) ----------
for alpha in [0.1, 1.0, 10.0]:
    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    lasso_weights = pd.DataFrame({'Feature': X.columns, 'Weight': lasso_model.coef_})
    top_lasso = lasso_weights.reindex(lasso_weights.Weight.abs().sort_values(ascending=False).index).head(5)
    print(f"\nLasso Regression (α = {alpha}):")
    print(top_lasso.to_string(index=False))

# ---------- Polynomial Regression (degree = 2) ----------
poly2_weights = pd.DataFrame({'Feature': feature_names_poly2, 'Weight': poly2_lr.coef_})
top_poly2 = poly2_weights.reindex(poly2_weights.Weight.abs().sort_values(ascending=False).index).head(5)
print("\nPolynomial Regression (Degree = 2):")
print(top_poly2.to_string(index=False))

# ---------- Polynomial Regression (degree = 3) ----------
poly3_weights = pd.DataFrame({'Feature': feature_names_poly3, 'Weight': poly3_lr.coef_})
top_poly3 = poly3_weights.reindex(poly3_weights.Weight.abs().sort_values(ascending=False).index).head(5)
print("\nPolynomial Regression (Degree = 3):")
print(top_poly3.to_string(index=False))
