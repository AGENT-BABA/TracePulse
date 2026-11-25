"""Log Parser and Feature Engineering Pipeline for LogSentry.

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
    r"(?P<level>\w+)\s*\|\s*"
    r"(?P<service>[\w-]+)\s*\|\s*"
    r"(?P<endpoint>[/\w.-]+)\s*\|\s*"
    r"(?P<latency_ms>[\d.]+)\s*\|\s*"
    r"(?P<status_code>\d+)\s*\|\s*"
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


class LogParser:
    """Parses raw text logs and creates normalized analytical features for ML anomaly detection."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    @staticmethod
    def normalize_message_template(message: str) -> str:
        """Normalize dynamic identifiers (IDs, IPs, numbers, UUIDs) into template tokens.

        Allows measuring the frequency and statistical rarity of distinct log event templates.
        """
        # Replace IP addresses
        t = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", message)
        # Replace UUIDs
        t = re.sub(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
            "<UUID>",
            t,
        )
        # Replace hex / memory addresses
        t = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", t)
        # Replace specific entity patterns like cart_1234, user_1234, txn_1234, order_1234
        t = re.sub(r"\b(cart|user|txn|order|vault|intent|sku|sku-|res)_[0-9a-zA-Z]+\b", r"<\1_ID>", t, flags=re.IGNORECASE)
        # Replace standalone numbers
        t = re.sub(r"\b\d+\b", "<NUM>", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def parse_lines(self, lines: List[str]) -> pd.DataFrame:
        """Parse raw log strings into a structured DataFrame.

        Skips unparseable lines gracefully without failing the entire batch.
        """
        records: List[dict] = []
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = LOG_PATTERN.match(line)
            if match:
                d = match.groupdict()
                records.append(
                    {
                        "raw_line": line,
                        "line_number": line_num + 1,
                        "timestamp_str": d["timestamp"],
                        "level": d["level"].upper(),
                        "service": d["service"],
                        "endpoint": d["endpoint"],
                        "latency_ms": float(d["latency_ms"]),
                        "status_code": int(d["status_code"]),
                        "message": d["message"],
                    }
                )

        if not records:
            raise ValueError("No valid structured log records found in input.")

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(
            df["timestamp_str"].str.replace(",", "."), format="%Y-%m-%d %H:%M:%S.%f"
        )
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Engineer temporal, rolling-window, and statistical features for Isolation Forest.

        Returns:
            Tuple of (augmented_dataframe, scaled_feature_matrix)
        """
        df = df.copy()

        # 1. Direct point features
        df["is_error"] = (df["status_code"] >= 400).astype(int)
        df["severity_score"] = df["level"].map(lambda l: SEVERITY_MAP.get(l, 1)).astype(int)

        # 2. Rolling aggregations per service (trailing 20 logs)
        # Grouped rolling calculation maintaining index alignment
        df["rolling_error_rate"] = (
            df.groupby("service")["is_error"]
            .rolling(window=20, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

        df["rolling_latency_mean"] = (
            df.groupby("service")["latency_ms"]
            .rolling(window=20, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(df["latency_ms"])
        )

        # 3. Service response time z-score
        service_stats = df.groupby("service")["latency_ms"].agg(["mean", "std"]).reset_index()
        service_stats["std"] = service_stats["std"].fillna(1.0).replace(0.0, 1.0)
        stats_map = service_stats.set_index("service").to_dict(orient="index")

        def calc_zscore(row: pd.Series) -> float:
            srv = row["service"]
            st = stats_map.get(srv, {"mean": 50.0, "std": 20.0})
            return (row["latency_ms"] - st["mean"]) / st["std"]

        df["latency_zscore"] = df.apply(calc_zscore, axis=1).clip(-5.0, 25.0)

        # 4. Normalized template rarity (inverse log-frequency)
        df["template"] = df["message"].apply(self.normalize_message_template)
        template_counts = df["template"].value_counts().to_dict()
        total_logs = len(df)

        # Rarity score: higher value indicates rarer, less frequent log template
        df["template_rarity"] = df["template"].apply(
            lambda t: np.log((total_logs + 1.0) / (template_counts.get(t, 1) + 1.0))
        )

        # Extract numerical feature matrix
        feature_matrix = df[FEATURE_COLUMNS].values
        scaled_features = self.scaler.fit_transform(feature_matrix)

        return df, scaled_features
