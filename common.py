"""
Code shared by train.py and app.py.

Why a separate module: app.py must rebuild the exact same features and model
class that train.py used, but it must never *run* training. If app.py imported
train.py, Python would execute the whole training script on import. Putting the
shared pieces here avoids that.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart_2022_sample.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET = "HadHeartAttack"
SEED = 42

# ---------------------------------------------------------------------------
# YOUR DECISIONS GO HERE
# ---------------------------------------------------------------------------
# Columns to exclude from the model. Every entry needs a reason you can defend
# in REPORT.md (e.g. "only known after the event we're trying to detect").
# Empty by default, so the starter pipeline uses everything.
DROP_COLUMNS: list[str] = []


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw survey columns into model features. THIS IS YOUR 30%.

    Receives the raw dataframe (target included) and returns a new one.
    Right now it does nothing, so the pipeline runs end to end with plain
    one-hot encoding of every text column. That's your "before" baseline:
    commit it, record its scores, then add features one at a time and
    keep a short log of what each change did to validation PR-AUC.

    Rules of thumb for anything you add here:
    - It must work row by row, using no statistics computed from the whole
      dataset (those belong in the sklearn preprocessor, which is fit on the
      training split only). Otherwise test information leaks into training.
    - Any new column you create must be numeric (int/float) if you want it
      scaled as a number, or text if you want it one-hot encoded.
    - Drop the raw column if your new column replaces it, or you'll feed the
      model the same information twice.
    """
    df = df.copy()
    # e.g. df["my_feature"] = ...
    return df


# ---------------------------------------------------------------------------
# Plumbing: you shouldn't need to change much below this line
# ---------------------------------------------------------------------------
def make_xy(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw rows -> (feature dataframe, 0/1 target). Used identically by train and app."""
    df = engineer_features(raw)
    y = (df[TARGET] == "Yes").astype(int).to_numpy()
    X = df.drop(columns=[TARGET, *DROP_COLUMNS])
    return X, y


def column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric columns get scaled; everything else is treated as categorical."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Impute -> encode -> standardize everything.

    Standardizing *all* columns (one-hot ones included) gives gradient descent
    a well-conditioned problem, which is what lets the two PyTorch versions
    converge to the same answer as sklearn. Coefficients are then "change in
    log-odds per 1 standard deviation" for every feature.

    The imputation strategies are generic defaults. If you use the
    with_nans file, deciding how to treat missing answers is a real
    feature-engineering choice worth revisiting.
    """
    encode = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # drop="first" avoids the dummy-variable trap (k one-hot
                        # columns + an intercept are perfectly collinear).
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                 sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("encode", encode), ("scale", StandardScaler())])


class LogisticRegressionNet(nn.Module):
    """Logistic regression as a PyTorch module: one linear layer.

    forward() returns logits (raw scores), not probabilities. The sigmoid is
    folded into BCEWithLogitsLoss during training and applied explicitly at
    prediction time.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
