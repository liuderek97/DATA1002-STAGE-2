import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


dfA = pd.read_csv("dataset_A_gdp_life.csv")
dfB = pd.read_csv("dataset_B_gdp_co2_life.csv")
dfC = pd.read_csv("dataset_C_gdp_co2_year_life.csv")

datasets = {"A": dfA, "B": dfB, "C": dfC}

def train_and_plot(tag, df):
    # Clean
    df = df.dropna(subset=["Life Expectancy", "GDP per capita USD", "Year"])
    if "CO2 Use" in df.columns:
        df = df.dropna(subset=["CO2 Use"])
    
    # Transform
    df["log_GDP_per_capita"] = np.log(df["GDP per capita USD"])
    features = ["log_GDP_per_capita", "Year"]
    if "CO2 Use" in df.columns:
        features.append("CO2 Use")

    X = df[features]
    y = df["Life Expectancy"]

    # Split 75/15/10
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)
    X_trainval, y_trainval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])

    # Models
    models = {
        "Linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
        "RF": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_trainval, y_trainval)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MSE": mse, "R2": r2}

        # Save plot as PNG
        plt.figure(figsize=(5,5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.title(f"{tag} - {name}\nLife Expectancy\nR²={r2:.2f}, MSE={mse:.2f}")
        plt.xlabel("Actual Life Expectancy")
        plt.ylabel("Predicted Life Expectancy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{tag}_{name.lower()}_life_expectancy.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Print summary
    best_model = min(results, key=lambda x: results[x]["MSE"])
    print(f"\nDataset {tag} Results:")
    for name, m in results.items():
        print(f"  {name:<10} | MSE={m['MSE']:.3f} | R²={m['R2']:.3f}")
    print(f"  → Best model: {best_model}\n")

# ---- Run for all datasets ----
for tag, df in datasets.items():
    train_and_plot(tag, df)
