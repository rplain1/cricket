import polars as pl
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path

OUT_DIR = Path("models/")
df = pl.read_csv("report/q3a.csv")

model = LinearRegression()

X = df["over"].to_numpy().reshape(-1, 1)
y = df["runs"].to_numpy()

model.fit(X, y)

joblib.dump(model, OUT_DIR / "linear_model.pkl")