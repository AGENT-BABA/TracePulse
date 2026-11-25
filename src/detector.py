"""Unsupervised Anomaly Detector for LogSentry.

Implements Isolation Forest anomaly detection with severity scoring,
contamination thresholding, and latency profiling.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class LogAnomalyDetector:
    """Unsupervised anomaly detector leveraging Isolation Forest on multi-dimensional telemetry."""

    def __init__(
        self,
        contamination: float = 0.03,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        """Initialize detector with tuning parameters."""
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: IsolationForest | None = None

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_matrix: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit Isolation Forest model and assign anomaly labels and scores.

        Args:
            df: Parsed log DataFrame.
            feature_matrix: Scaled numerical feature array.

        Returns:
            Tuple of (annotated_dataframe, detection_metrics_dict)
        """
        start_time = time.perf_counter()

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=-1,
        )

        # anomaly_labels: -1 for anomaly, 1 for inlier
        raw_labels = self.model.fit_predict(feature_matrix)

        # decision_function: lower/more negative means more anomalous
        raw_scores = self.model.decision_function(feature_matrix)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        df_out = df.copy()
        df_out["anomaly_label"] = raw_labels
        df_out["is_anomaly"] = df_out["anomaly_label"] == -1

        # Anomaly severity score: inverted and normalized so higher score (0.0 to 1.0)
        # represents greater severity/anomaly strength
        # raw_scores typically range from ~ -0.4 (extreme anomaly) to +0.25 (typical normal)
        min_score = float(np.min(raw_scores))
        max_score = float(np.max(raw_scores))
        score_range = max(max_score - min_score, 1e-6)

        # Invert: lowest raw score -> anomaly_severity = 1.0; highest raw score -> 0.0
        df_out["raw_decision_score"] = raw_scores
        df_out["anomaly_severity"] = np.round((max_score - raw_scores) / score_range, 4)

        total_logs = len(df_out)
        anomalies_detected = int((df_out["is_anomaly"]).sum())
        anomaly_rate_pct = round((anomalies_detected / total_logs) * 100, 2) if total_logs > 0 else 0.0

        metrics: Dict[str, Any] = {
            "total_logs": total_logs,
            "anomalies_detected": anomalies_detected,
            "normal_logs": total_logs - anomalies_detected,
            "anomaly_rate_pct": anomaly_rate_pct,
            "contamination_parameter": self.contamination,
            "n_estimators": self.n_estimators,
            "inference_time_ms": inference_time_ms,
        }

        return df_out, metrics
