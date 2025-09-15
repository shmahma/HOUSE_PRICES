import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

df = pd.read_csv("data/train.csv")
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # <-- important
])


preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=500, random_state=42),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001)
}

results = {}
for name, reg in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    results[name] = rmse
    print(f"{name} RMSE: {rmse:.2f}")

best_model = min(results, key=results.get)
print("\n Best Model :", best_model, " RMSE =", results[best_model])

model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', models[best_model])
    ])

model.fit(X, y)

joblib.dump(model, "model.pkl")
print(f"Model pipeline saved: model.pkl")
