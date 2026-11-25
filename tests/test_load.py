"""High-Volume Stress and Load Benchmark Test for TracePulse.

Generates 50,000 realistic log lines (~5-7 MB) with injected anomaly clusters,
executes the full ingestion, feature engineering, Isolation Forest detection,
and incident RCA pipeline, measuring throughput (logs/sec) and latency against
strict SLAs.
"""

from __future__ import annotations

import time
import tracemalloc
import pytest

from src.detector import LogAnomalyDetector
from src.log_generator import SyntheticLogGenerator
from src.parser import LogParser
from src.rca_engine import IncidentGrouper, RCAEngine


@pytest.fixture(scope="module")
def sample_50k_logs():
    """Generate 50,000 logs once per module with reproducible seed."""
    gen = SyntheticLogGenerator(seed=1337)
    return gen.generate_logs(normal_count=50000)


class TestHighVolumeLoadBenchmark:
    """Stress tests the local pipeline at 50,000 logs and asserts SLA thresholds."""

    def test_end_to_end_50k_load_benchmark(self, sample_50k_logs):
        """Benchmark ingestion, feature engineering, ML detection, and RCA at scale."""
        total_logs = len(sample_50k_logs)
        assert total_logs >= 50000, f"Expected at least 50k logs, got {total_logs}"

        tracemalloc.start()
        pipeline_start = time.perf_counter()

        # ---------------- Stage 1: Parsing ----------------
        t0 = time.perf_counter()
        parser = LogParser()
        df_parsed = parser.parse_lines(sample_50k_logs)
        parse_time = time.perf_counter() - t0

        # ---------------- Stage 2: Feature Engineering ----------------
        t1 = time.perf_counter()
        df_featured, feature_matrix = parser.engineer_features(df_parsed)
        feat_time = time.perf_counter() - t1

        parse_plus_feat_time = parse_time + feat_time

        # ---------------- Stage 3: Isolation Forest ML ----------------
        t2 = time.perf_counter()
        detector = LogAnomalyDetector(contamination=0.02, n_estimators=100, random_state=42)
        df_results, metrics = detector.fit_predict(df_featured, feature_matrix)
        ml_time = time.perf_counter() - t2

        # ---------------- Stage 4: Incident Clustering & RCA ----------------
        t3 = time.perf_counter()
        anomalies_df = df_results[df_results["is_anomaly"]]
        incidents = IncidentGrouper.group_anomalies(anomalies_df, window_seconds=60)
        rca_engine = RCAEngine()
        reports = rca_engine.analyze_all_incidents(incidents, force_heuristic=True)
        rca_time = time.perf_counter() - t3

        total_pipeline_time = time.perf_counter() - pipeline_start
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem / (1024 * 1024)
        throughput_logs_sec = round(total_logs / max(total_pipeline_time, 0.001), 1)

        # Print Benchmark Table to stdout
        print("\n" + "=" * 70)
        print(f"🚀 TRACEPULSE HIGH-VOLUME LOAD BENCHMARK (DATASET: {total_logs:,} LOGS)")
        print("=" * 70)
        print(f"{'Pipeline Stage':<35} | {'Latency':<12} | {'Throughput':<16}")
        print("-" * 70)
        print(f"{'1. Log Line Regex Parsing':<35} | {parse_time:.3f}s     | {int(total_logs/parse_time):,} logs/s")
        print(f"{'2. Feature Engineering & Scaling':<35} | {feat_time:.3f}s     | {int(total_logs/feat_time):,} logs/s")
        print(f"{'   -> Combined Parse + Features':<35} | {parse_plus_feat_time:.3f}s     | {int(total_logs/parse_plus_feat_time):,} logs/s")
        print(f"{'3. IsolationForest (Fit + Predict)':<35} | {ml_time:.3f}s     | {int(total_logs/ml_time):,} logs/s")
        print(f"{'4. Incident Clustering & Heuristic RCA':<35} | {rca_time:.3f}s     | {int(total_logs/rca_time):,} logs/s")
        print("-" * 70)
        print(f"{'TOTAL PIPELINE END-TO-END':<35} | {total_pipeline_time:.3f}s     | {throughput_logs_sec:,} logs/s")
        print(f"{'Peak Heap Memory Utilized':<35} | {peak_mem_mb:.2f} MB")
        print(f"{'Anomalies Detected / Clustered':<35} | {metrics['anomalies_detected']:,} anomalies -> {len(incidents)} incidents")
        print("=" * 70 + "\n")

        # ---------------- Strict SLA Assertions ----------------
        # 1. Parsing + Feature engineering SLA: < 3.0 seconds
        assert parse_plus_feat_time < 3.0, (
            f"Parsing + feature engineering exceeded SLA: {parse_plus_feat_time:.2f}s >= 3.0s"
        )

        # 2. ML Isolation Forest fit + inference SLA: < 2.0 seconds
        assert ml_time < 2.0, (
            f"IsolationForest inference exceeded SLA: {ml_time:.2f}s >= 2.0s"
        )

        # 3. Incident clustering + RCA SLA: < 1.0 second
        assert rca_time < 1.0, (
            f"Clustering and RCA exceeded SLA: {rca_time:.2f}s >= 1.0s"
        )

        # 4. Total pipeline SLA: < 6.0 seconds
        assert total_pipeline_time < 6.0, (
            f"Total pipeline time exceeded SLA: {total_pipeline_time:.2f}s >= 6.0s"
        )

        # 5. Incident clustering must have identified the anomalies
        assert len(incidents) >= 3, f"Expected at least 3 incident clusters, found {len(incidents)}"

    def test_memory_stability_across_consecutive_batches(self):
        """Verify memory does not leak across 5 consecutive 10,000-log batch runs."""
        parser = LogParser()
        detector = LogAnomalyDetector(contamination=0.03, n_estimators=50, random_state=42)
        gen = SyntheticLogGenerator(seed=99)
        batch = gen.generate_logs(normal_count=10000)

        tracemalloc.start()
        memory_checkpoints = []

        for _ in range(5):
            df_parsed = parser.parse_lines(batch)
            df_featured, feature_matrix = parser.engineer_features(df_parsed)
            df_results, _ = detector.fit_predict(df_featured, feature_matrix)
            anomalies_df = df_results[df_results["is_anomaly"]]
            incidents = IncidentGrouper.group_anomalies(anomalies_df, window_seconds=60)
            rca_engine = RCAEngine()
            _ = rca_engine.analyze_all_incidents(incidents, force_heuristic=True)

            cur_mem, _ = tracemalloc.get_traced_memory()
            memory_checkpoints.append(cur_mem / (1024 * 1024))

        tracemalloc.stop()

        # Growth from run 1 to run 5 should be negligible (no runaway memory leak)
        growth_mb = memory_checkpoints[-1] - memory_checkpoints[0]
        assert growth_mb < 25.0, f"Memory growth between consecutive runs too high: {growth_mb:.2f} MB"
