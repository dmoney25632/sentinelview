"""Machine-learning anomaly detection using scikit-learn Isolation Forest."""

import pickle
from typing import ClassVar

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

VESSEL_FEATURES: list[str] = [
    "computed_speed_kts",
    "heading_delta",
    "time_gap_s",
    "distance_km",
    "loitering",
    "sog_vs_computed_delta",
]

FLIGHT_FEATURES: list[str] = [
    "computed_speed_kts",
    "heading_delta",
    "time_gap_s",
    "distance_km",
    "altitude_change_ft",
    "descent_rate_fpm",
]


class MLAnomalyDetector:
    """Isolation Forest anomaly detector for vessel or flight track data.

    Args:
        contamination: Expected proportion of anomalies in training data.
        mode: ``"vessel"`` or ``"flight"`` — selects the feature set to use.
    """

    _FEATURE_MAP: ClassVar[dict[str, list[str]]] = {
        "vessel": VESSEL_FEATURES,
        "flight": FLIGHT_FEATURES,
    }

    def __init__(self, contamination: float = 0.05, mode: str = "vessel") -> None:
        if mode not in self._FEATURE_MAP:
            raise ValueError(f"mode must be 'vessel' or 'flight', got {mode!r}")
        self.mode = mode
        self.contamination = contamination
        self._features = self._FEATURE_MAP[mode]
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "MLAnomalyDetector":
        """Fit the scaler and Isolation Forest on clean rows of *df*.

        Args:
            df: DataFrame containing at least the feature columns for the
                selected mode.  Rows with NaN in any feature column are
                dropped before fitting.

        Returns:
            Self, to allow method chaining.
        """
        X = df[self._features].dropna()
        self._scaler.fit(X)
        self._model.fit(self._scaler.transform(X))
        self.fitted = True
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every row in *df* and add anomaly columns.

        Args:
            df: DataFrame containing at least the feature columns for the
                selected mode.

        Returns:
            Copy of *df* with three new columns:

            - ``ml_raw_score``: negated Isolation Forest ``score_samples``
              output (higher value = more anomalous).
            - ``ml_anomaly_score``: min-max normalised ``ml_raw_score``
              in the range ``[0, 1]``.
            - ``ml_flag``: ``1`` when the model predicts ``-1`` (outlier),
              ``0`` otherwise.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if not self.fitted:
            raise RuntimeError("MLAnomalyDetector must be fitted before calling score().")

        out = df.copy()
        X = out[self._features].fillna(0)
        X_scaled = self._scaler.transform(X)

        raw = -self._model.score_samples(X_scaled)
        out["ml_raw_score"] = raw

        lo, hi = raw.min(), raw.max()
        if hi > lo:
            out["ml_anomaly_score"] = (raw - lo) / (hi - lo)
        else:
            out["ml_anomaly_score"] = 0.0

        preds = self._model.predict(X_scaled)
        out["ml_flag"] = (preds == -1).astype(int)

        return out

    def save(self, path: str) -> None:
        """Persist the scaler and model to a pickle file.

        Args:
            path: File system path to write the pickle to.
        """
        with open(path, "wb") as fh:
            pickle.dump({"scaler": self._scaler, "model": self._model, "mode": self.mode}, fh)

    @classmethod
    def load(cls, path: str) -> "MLAnomalyDetector":
        """Load a previously saved detector from a pickle file.

        Args:
            path: File system path of the pickle to read.

        Returns:
            A fitted :class:`MLAnomalyDetector` instance.
        """
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        instance = cls(mode=payload["mode"])
        instance._scaler = payload["scaler"]
        instance._model = payload["model"]
        instance.fitted = True
        return instance
