"""Edge-Case and Resilience Test Suite for TracePulse.

Validates the parser, ML anomaly detector, incident grouper, and RCA engine
against hostile inputs, boundary conditions, zero-variance streams,
and upstream LLM API failures.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.detector import LogAnomalyDetector
from src.parser import LogParser
from src.rca_engine import IncidentGrouper, IncidentReport, IncidentWindow, RCAEngine


class TestEmptyAndWhitespaceInputs:
    """Verifies that empty and whitespace-only inputs do not crash the pipeline."""

    def test_empty_log_lines_returns_empty_dataframe(self):
        parser = LogParser()
        df = parser.parse_lines([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "timestamp" in df.columns

    def test_whitespace_only_log_lines(self):
        parser = LogParser()
        lines = ["   ", "\n", "\t\t", "   \r\n   ", "# comment only"]
        df = parser.parse_lines(lines)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_feature_engineering_on_empty_dataframe(self):
        parser = LogParser()
        empty_df = pd.DataFrame()
        df_feat, matrix = parser.engineer_features(empty_df)
        assert df_feat.empty
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] == 0

    def test_detector_on_empty_features(self):
        detector = LogAnomalyDetector()
        df_empty = pd.DataFrame()
        empty_matrix = np.empty((0, 7))
        df_out, metrics = detector.fit_predict(df_empty, empty_matrix)
        assert df_out.empty
        assert metrics["total_logs"] == 0
        assert metrics["anomalies_detected"] == 0

    def test_incident_grouper_on_empty_dataframe(self):
        incidents = IncidentGrouper.group_anomalies(pd.DataFrame())
        assert incidents == []


class TestMalformedAndCorruptedInputs:
    """Verifies robustness against corrupted, truncated, and hostile log lines."""

    def test_mixed_valid_and_malformed_lines(self):
        parser = LogParser()
        lines = [
            "THIS IS NOT A VALID LOG LINE",
            "2026-11-25 09:00:00,100 | INFO | auth-service | /api/v1/auth | 45.20 | 200 | Success",
            "CORRUPTED | MISSING | FIELDS",
            "2026-99-99 99:99:99,999 | WRONG | auth-service | /api/v1 | INVALID_NUM | 200 | Broken",
            "\x00\xff\xfe\x01\x02 BINARY GARBAGE",
            "2026-11-25 09:00:01,200 | ERROR | order-api | /api/v1/orders | 3500.00 | 503 | Timeout",
        ]
        df = parser.parse_lines(lines)
        assert len(df) == 2
        assert set(df["service"]) == {"auth-service", "order-api"}

    def test_sql_and_xss_injection_in_message(self):
        parser = LogParser()
        lines = [
            "2026-11-25 09:00:00,100 | INFO | auth-service | /api/v1/login | 50.00 | 200 | "
            "User login query: ' OR '1'='1' -- DROP TABLE users;",
            "2026-11-25 09:00:01,200 | WARN | auth-service | /api/v1/profile | 65.00 | 200 | "
            "<script>alert('XSS_ATTACK');</script><img src=x onerror=alert(1)>",
        ]
        df = parser.parse_lines(lines)
        assert len(df) == 2
        df_feat, matrix = parser.engineer_features(df)
        assert matrix.shape[0] == 2
        assert not np.isnan(matrix).any()


class TestSingleRowAndTinyInputs:
    """Verifies that tiny inputs (< 20 rows) do not propagate NaN into IsolationForest."""

    def test_single_log_line_pipeline(self):
        parser = LogParser()
        lines = [
            "2026-11-25 09:00:00,100 | INFO | auth-service | /api/v1/auth | 45.20 | 200 | Single entry"
        ]
        df = parser.parse_lines(lines)
        assert len(df) == 1

        df_feat, matrix = parser.engineer_features(df)
        assert matrix.shape == (1, 7)
        assert not np.isnan(matrix).any()
        assert not np.isinf(matrix).any()

        detector = LogAnomalyDetector(contamination=0.10, n_estimators=50)
        df_out, metrics = detector.fit_predict(df_feat, matrix)
        assert len(df_out) == 1
        assert "is_anomaly" in df_out.columns
        assert not np.isnan(df_out["anomaly_severity"].iloc[0])

    def test_two_log_lines_pipeline(self):
        parser = LogParser()
        lines = [
            "2026-11-25 09:00:00,100 | INFO | auth-service | /api/v1/auth | 45.00 | 200 | Token valid",
            "2026-11-25 09:00:01,200 | ERROR | auth-service | /api/v1/auth | 450.00 | 500 | Crash",
        ]
        df = parser.parse_lines(lines)
        df_feat, matrix = parser.engineer_features(df)
        assert matrix.shape == (2, 7)
        assert not np.isnan(matrix).any()

        detector = LogAnomalyDetector(contamination=0.10, n_estimators=50)
        df_out, metrics = detector.fit_predict(df_feat, matrix)
        assert len(df_out) == 2


class TestUniformAndZeroVarianceLogs:
    """Verifies that uniform logs with zero variance across features do not cause division-by-zero."""

    def test_500_identical_logs(self):
        parser = LogParser()
        # 500 perfectly identical log entries
        base_line = "2026-11-25 09:00:00,000 | INFO | auth-service | /api/v1/test | 50.00 | 200 | Steady baseline"
        lines = [base_line for _ in range(500)]

        df = parser.parse_lines(lines)
        assert len(df) == 500

        df_feat, matrix = parser.engineer_features(df)
        assert matrix.shape == (500, 7)
        assert not np.isnan(matrix).any()
        assert not np.isinf(matrix).any()

        detector = LogAnomalyDetector(contamination=0.03, n_estimators=50)
        df_out, metrics = detector.fit_predict(df_feat, matrix)
        assert len(df_out) == 500
        # All severity values must be finite and normalized between 0.0 and 1.0
        assert not df_out["anomaly_severity"].isna().any()
        assert df_out["anomaly_severity"].min() >= 0.0
        assert df_out["anomaly_severity"].max() <= 1.0


class TestAllAnomalyFlood:
    """Verifies that an all-anomaly flood groups into compact incidents without spamming."""

    def test_100_percent_error_flood(self):
        parser = LogParser()
        start = datetime(2026, 11, 25, 10, 0, 0)
        lines = []
        for i in range(50):
            t = start + timedelta(seconds=i * 2)  # within 60s window
            ts = t.strftime("%Y-%m-%d %H:%M:%S,") + f"{int(t.microsecond/1000):03d}"
            lines.append(
                f"{ts} | CRITICAL | order-api | /api/v1/orders | 7500.00 | 503 | "
                f"Timeout waiting for connection from pool: HikariPool-1 (wait_time=7200ms)"
            )

        df = parser.parse_lines(lines)
        df_feat, matrix = parser.engineer_features(df)

        detector = LogAnomalyDetector(contamination=0.10, n_estimators=50)
        df_out, _ = detector.fit_predict(df_feat, matrix)

        # Mark all as anomalous for grouping verification
        df_out["is_anomaly"] = True
        incidents = IncidentGrouper.group_anomalies(df_out, window_seconds=60)

        # Since all 50 events occurred within 2s of each other on order-api,
        # they must be clustered into exactly ONE incident window, not 50.
        assert len(incidents) == 1
        assert incidents[0].log_count == 50
        assert incidents[0].service == "order-api"


class TestLLMResilienceAndFallbacks:
    """Validates that RCAEngine gracefully handles OpenAI timeouts, 429s, bad keys, and malformed JSON."""

    @pytest.fixture
    def mock_incident(self):
        logs = pd.DataFrame([
            {
                "timestamp": datetime(2026, 11, 25, 10, 0, 0),
                "latency_ms": 6500.0,
                "status_code": 503,
                "message": "Timeout waiting for connection from pool: HikariPool-1 (active=50, idle=0)",
            },
            {
                "timestamp": datetime(2026, 11, 25, 10, 0, 5),
                "latency_ms": 7100.0,
                "status_code": 503,
                "message": "Database pool exhausted, request timed out",
            },
        ])
        return IncidentWindow("INC-001", "order-api", logs)

    def test_missing_or_placeholder_api_key_activates_fallback(self, mock_incident):
        rca = RCAEngine(api_key="your_api_key_here")
        assert not rca.is_api_configured()
        report = rca.diagnose_incident(mock_incident)
        assert isinstance(report, IncidentReport)
        assert report.incident_id == "INC-001"
        assert report.severity == "CRITICAL"
        assert "Database Connection Pool" in report.incident_title
        assert len(report.remediation_steps) == 3

    def test_timeout_error_falls_back_cleanly(self, mock_incident):
        rca = RCAEngine(api_key="sk-valid-looking-key-for-test-1234567890")
        assert rca.is_api_configured()

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.beta.chat.completions.parse.side_effect = TimeoutError("Connection to OpenAI timed out")
            mock_client.chat.completions.create.side_effect = TimeoutError("Connection to OpenAI timed out")
            mock_openai_cls.return_value = mock_client

            report = rca.diagnose_incident(mock_incident)
            assert isinstance(report, IncidentReport)
            assert report.incident_id == "INC-001"
            assert "Database Connection Pool" in report.incident_title

    def test_rate_limit_429_error_falls_back_cleanly(self, mock_incident):
        rca = RCAEngine(api_key="sk-valid-looking-key-for-test-1234567890")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            # Simulate 429 RateLimitError
            mock_client.beta.chat.completions.parse.side_effect = Exception("429 Rate limit exceeded. Try again in 20s.")
            mock_client.chat.completions.create.side_effect = Exception("429 Rate limit exceeded.")
            mock_openai_cls.return_value = mock_client

            report = rca.diagnose_incident(mock_incident)
            assert isinstance(report, IncidentReport)
            assert report.severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            assert len(report.remediation_steps) == 3

    def test_malformed_json_response_falls_back_cleanly(self, mock_incident):
        rca = RCAEngine(api_key="sk-valid-looking-key-for-test-1234567890")

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.beta.chat.completions.parse.side_effect = Exception("Parse failed")

            # Return garbage non-JSON content
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "```text\nSorry, I cannot produce valid JSON right now.\n```"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            report = rca.diagnose_incident(mock_incident)
            assert isinstance(report, IncidentReport)
            assert report.incident_id == "INC-001"
            assert len(report.remediation_steps) == 3
