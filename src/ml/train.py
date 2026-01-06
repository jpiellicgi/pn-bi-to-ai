import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# SAMPLE ML CODE TO GET STARTED

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2] if __name__ != "__main__" else Path.cwd()
DATA_PATH = ROOT / "data" / "processed" / "vehicles_2017to2023.csv"

# ---- Load ----
df = pd.read_csv(DATA_PATH)

# ---- Minimal feature/target selection ----
# Features: TRAV_SP (numeric), MODEL (categorical)
# Target: is_make_ford (1 if MAKE == 'Ford', else 0)
required_cols = ["TRAV_SP", "MODEL", "MAKE"]
if not set(required_cols).issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

X = df[["TRAV_SP", "MODEL"]].copy()
y = (df["MAKE"].astype(str).str.strip().str.upper() == "FORD").astype(int)

# Drop rows with missing values in our small feature set
mask = X["TRAV_SP"].notna() & X["MODEL"].notna()
X, y = X[mask], y[mask]

# ---- Preprocess & model ----
numeric_features = ["TRAV_SP"]
categorical_features = ["MODEL"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = LogisticRegression(max_iter=1000)

clf = Pipeline(steps=[("prep", preprocess), ("model", model)])

# ---- Train / evaluate ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

print(f"Test accuracy: {acc:.3f}")
