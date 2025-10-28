# ==========================================
# Simple country-grouped train/val/test split
# Models: Linear, RandomForest, KNN
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----- Load data -----
life = pd.read_csv("Life Expectancy Data.csv")
co2  = pd.read_csv("co2_emission.csv")
gdp  = pd.read_csv("f5a9ad86-f7cb-42ba-868e-b444c1d52fa4_Data.csv")

# Clean and select
life = life[["Country", "Year", "Life expectancy "]].rename(columns={"Life expectancy ": "Life Expectancy"})
gdp = gdp.rename(columns={"Country Name":"Country","Time":"Year","Value":"GDP per capita USD"})
co2 = co2.rename(columns={"Entity":"Country","Annual CO₂ emissions (tonnes per capita)":"CO2 per capita"})

# Merge
df = life.merge(gdp, on=["Country","Year"], how="inner").merge(co2, on=["Country","Year"], how="inner")
df = df.dropna().copy()
df["log_GDP_per_capita"] = np.log(df["GDP per capita USD"])

# ----- Grouped split by Country -----
seed = 42
groups = df["Country"]

# 75% train, 25% temp
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
train_idx, temp_idx = next(gss1.split(df, groups=groups))
train, temp = df.iloc[train_idx], df.iloc[temp_idx]

# 15% val, 10% test from 25% temp
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
val_idx, test_idx = next(gss2.split(temp, groups=temp["Country"]))
val, test = temp.iloc[val_idx], temp.iloc[test_idx]

# ----- Prepare data -----
X_train = train[["log_GDP_per_capita","CO2 per capita","Year"]]
y_train = train["Life Expectancy"]
X_val   = val[["log_GDP_per_capita","CO2 per capita","Year"]]
y_val   = val["Life Expectancy"]
X_test  = test[["log_GDP_per_capita","CO2 per capita","Year"]]
y_test  = test["Life Expectancy"]

# ----- Define models -----
models = {
    "Linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
    "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=seed)
}

# ----- Train and validate -----
val_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    mse = mean_squared_error(y_val, pred_val)
    r2  = r2_score(y_val, pred_val)
    val_scores[name] = mse
    print(f"{name:12s} -> Val MSE: {mse:.3f}, Val R²: {r2:.3f}")

best = min(val_scores, key=val_scores.get)
print(f"\nBest model (validation): {best}")

# ----- Final test -----
final_model = models[best]
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_model.fit(X_trainval, y_trainval)
pred_test = final_model.predict(X_test)
print(f"\nTEST RESULTS ({best})")
print("MSE:", mean_squared_error(y_test, pred_test))
print("R² :", r2_score(y_test, pred_test))
