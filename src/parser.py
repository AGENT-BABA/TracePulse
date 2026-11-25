"""Log Parser and Feature Engineering Pipeline for TracePulse.

Extracts structured telemetry from raw log streams and engineers statistical,
temporal, and message-rarity features for unsupervised anomaly detection.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Regex pattern to parse structured logs:
# YYYY-MM-DD HH:MM:SS,mmm | LEVEL | SERVICE | ENDPOINT | LATENCY_MS | STATUS_CODE | MESSAGE
LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*\|\s*"
    r"(?P<level>[A-Za-z]+)\s*\|\s*"
    r"(?P<service>[A-Za-z0-9_-]+)\s*\|\s*"
    r"(?P<endpoint>[/\w.-]+)\s*\|\s*"
    r"(?P<latency_ms>[\d.]+)\s*\|\s*"
    r"(?P<status_code>\d{3})\s*\|\s*"
    r"(?P<message>.*)$"
)

SEVERITY_MAP: Dict[str, int] = {
    "DEBUG": 0,
    "INFO": 1,
    "WARN": 2,
    "WARNING": 2,
    "ERROR": 4,
    "CRITICAL": 5,
    "FATAL": 5,
}

FEATURE_COLUMNS = [
    "latency_ms",
    "is_error",
    "severity_score",
    "rolling_error_rate",
    "rolling_latency_mean",
    "latency_zscore",
    "template_rarity",
]

PARSED_COLUMNS = [
    "raw_line",
    "line_number",
    "timestamp_str",
    "level",
    "service",
    "endpoint",
    "latency_ms",
    "status_code",
    "message",
    "timestamp",
]

# Precompiled regexes for high-performance message template normalization
RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
RE_HEX = re.compile(r"0x[0-9a-fA-F]+")
RE_ENTITY = re.compile(r"\b(cart|user|txn|order|vault|intent|sku|sku-|res)_[0-9a-zA-Z]+\b", re.IGNORECASE)
RE_NUM = re.compile(r"\b\d+\b")
RE_WS = re.compile(r"\s+")


class LogParser:
    """Parses raw text logs and creates normalized analytical features for ML anomaly detection."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    @staticmethod
    def normalize_message_template(message: str) -> str:
        """Normalize dynamic identifiers into template tokens for message rarity analysis."""
        if not message:
            return "<EMPTY>"
        t = RE_IP.sub("<IP>", message)
        t = RE_UUID.sub("<UUID>", t)
        t = RE_HEX.sub("<HEX>", t)
        t = RE_ENTITY.sub(r"<\1_ID>", t)
        t = RE_NUM.sub("<NUM>", t)
        t = RE_WS.sub(" ", t).strip()
        return t

    def parse_lines(self, lines: List[str]) -> pd.DataFrame:
        """Parse raw log strings into a structured DataFrame.

        Skips unparseable, malformed, or blank lines gracefully without crashing.
        Returns an empty DataFrame with schema if no valid lines exist.
        """
        rows: List[tuple] = []
        for line_num, line in enumerate(lines, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            match = LOG_PATTERN.match(line_str)
            if match:
                g = match.groups()
                try:
                    latency = float(g[4])
                    status = int(g[5])
                except (ValueError, TypeError):
                    continue

                rows.append((
                    line_str,
                    line_num,
                    g[0],
                    g[1].upper(),
                    g[2],
                    g[3],
                    latency,
                    status,
                    g[6],
                ))

        if not rows:
            df_empty = pd.DataFrame(columns=PARSED_COLUMNS)
            df_empty["timestamp"] = pd.to_datetime(df_empty["timestamp"])
            return df_empty

        df = pd.DataFrame(
            rows,
            columns=[
                "raw_line",
                "line_number",
                "timestamp_str",
                "level",
                "service",
                "endpoint",
                "latency_ms",
                "status_code",
                "message",
            ],
        )

        # Fast timestamp conversion
        ts_clean = df["timestamp_str"].str.replace(",", ".", regex=False)
        df["timestamp"] = pd.to_datetime(ts_clean, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        # Drop lines with unparseable timestamps if any
        if df["timestamp"].isna().any():
            df = df.dropna(subset=["timestamp"])

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Engineer temporal, rolling-window, and statistical features for Isolation Forest.

        Returns:
            Tuple of (augmented_dataframe, scaled_feature_matrix)
        """
        if df.empty:
            out_cols = PARSED_COLUMNS + FEATURE_COLUMNS + ["template"]
            empty_df = pd.DataFrame(columns=out_cols)
            empty_matrix = np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)
            return empty_df, empty_matrix

        df = df.copy()

        # 1. Point telemetry features
        df["is_error"] = (df["status_code"] >= 400).astype(int)
        df["severity_score"] = df["level"].map(lambda l: SEVERITY_MAP.get(l, 1)).astype(int)

        # 2. Vectorized rolling aggregations per microservice
        # Using transform with rolling maintains the original index directly
        df["rolling_error_rate"] = (
            df.groupby("service")["is_error"]
            .transform(lambda s: s.rolling(window=20, min_periods=1).mean())
            .fillna(0.0)
        )

        df["rolling_latency_mean"] = (
            df.groupby("service")["latency_ms"]
            .transform(lambda s: s.rolling(window=20, min_periods=1).mean())
            .fillna(df["latency_ms"])
        )

        # 3. Vectorized service response time z-score
        svc_mean = df.groupby("service")["latency_ms"].transform("mean")
        svc_std = df.groupby("service")["latency_ms"].transform("std").fillna(1.0).replace(0.0, 1.0)
        df["latency_zscore"] = ((df["latency_ms"] - svc_mean) / svc_std).fillna(0.0).clip(-5.0, 25.0)

        # 4. Message template rarity (compute on unique messages for speed, then map)
        unique_msgs = df["message"].unique()
        template_map = {m: self.normalize_message_template(m) for m in unique_msgs}
        df["template"] = df["message"].map(template_map)

        template_counts = df["template"].value_counts().to_dict()
        total_logs = len(df)
        df["template_rarity"] = df["template"].map(
            lambda t: float(np.log((total_logs + 1.0) / (template_counts.get(t, 1) + 1.0)))
        )

        # 5. Extract and standardize numerical feature matrix
        feature_matrix = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        # Guarantee no NaN or Inf
        np.nan_to_num(feature_matrix, copy=False, nan=0.0, posinf=25.0, neginf=-5.0)

        scaled_features = self.scaler.fit_transform(feature_matrix)

        return df, scaled_features
