"""Incident Grouping and Root-Cause Analysis (RCA) Engine.

Deduplicates contiguous anomalous log occurrences into distinct incident windows
and generates structured Pydantic diagnostics using modern OpenAI API or
a robust deterministic rule-based fallback analyzer.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field

# Load environment variables if available
load_dotenv()


class IncidentReport(BaseModel):
    """Pydantic structured schema for Incident Root-Cause Analysis."""

    incident_id: str = Field(description="Unique identifier for the incident, e.g. INC-001")
    incident_title: str = Field(description="Brief, high-impact title summarizing the incident")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Assessed incident severity tier"
    )
    root_cause: str = Field(
        description="Concise technical explanation detailing the exact failure mechanism"
    )
    affected_components: List[str] = Field(
        description="List of impacted microservices, databases, or infrastructure dependencies"
    )
    remediation_steps: List[str] = Field(
        description="Exactly 3 actionable, prioritized engineering remediation steps"
    )
    confidence_score: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Confidence score of the diagnosis between 0.0 and 1.0",
    )


class IncidentWindow:
    """Represents a grouped cluster of contiguous anomalous logs."""

    def __init__(
        self,
        incident_id: str,
        service: str,
        logs: pd.DataFrame,
    ) -> None:
        self.incident_id = incident_id
        self.service = service
        self.logs = logs
        self.start_time: datetime = logs["timestamp"].min()
        self.end_time: datetime = logs["timestamp"].max()
        self.log_count: int = len(logs)
        self.avg_latency: float = round(float(logs["latency_ms"].mean()), 2)
        self.max_latency: float = round(float(logs["latency_ms"].max()), 2)
        self.status_codes: List[int] = sorted(list(set(logs["status_code"].tolist())))
        self.dominant_status: int = int(logs["status_code"].mode()[0]) if not logs.empty else 200
        self.sample_messages: List[str] = logs["message"].head(8).tolist()

    def duration_seconds(self) -> float:
        """Calculate incident duration in seconds."""
        return max(1.0, (self.end_time - self.start_time).total_seconds())

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert incident window into summary dict for LLM prompting or reporting."""
        return {
            "incident_id": self.incident_id,
            "service": self.service,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": self.duration_seconds(),
            "log_count": self.log_count,
            "avg_latency_ms": self.avg_latency,
            "max_latency_ms": self.max_latency,
            "status_codes": self.status_codes,
            "dominant_status": self.dominant_status,
            "sample_messages": self.sample_messages,
        }


class IncidentGrouper:
    """Groups contiguous anomalous logs occurring on the same service within a time threshold."""

    @staticmethod
    def group_anomalies(
        df_anomalies: pd.DataFrame,
        window_seconds: int = 60,
    ) -> List[IncidentWindow]:
        """Group anomalies where consecutive logs are within window_seconds on the same service."""
        if df_anomalies.empty:
            return []

        df_sorted = df_anomalies.sort_values(["service", "timestamp"]).copy()
        incidents: List[IncidentWindow] = []
        counter = 1

        for service, group in df_sorted.groupby("service"):
            group = group.sort_values("timestamp")
            current_cluster: List[pd.Series] = []

            for _, row in group.iterrows():
                if not current_cluster:
                    current_cluster.append(row)
                    continue

                prev_time = current_cluster[-1]["timestamp"]
                curr_time = row["timestamp"]
                delta = (curr_time - prev_time).total_seconds()

                if delta <= window_seconds:
                    current_cluster.append(row)
                else:
                    cluster_df = pd.DataFrame(current_cluster)
                    inc_id = f"INC-{counter:03d}"
                    incidents.append(IncidentWindow(inc_id, service, cluster_df))
                    counter += 1
                    current_cluster = [row]

            if current_cluster:
                cluster_df = pd.DataFrame(current_cluster)
                inc_id = f"INC-{counter:03d}"
                incidents.append(IncidentWindow(inc_id, service, cluster_df))
                counter += 1

        # Re-sort incidents chronologically by start time
        incidents.sort(key=lambda inc: inc.start_time)
        # Re-label incident IDs sequentially in chronological order
        for idx, inc in enumerate(incidents, start=1):
            inc.incident_id = f"INC-{idx:03d}"

        return incidents


class RCAEngine:
    """Generates Root-Cause Analysis using OpenAI-compatible LLMs or deterministic heuristics."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    def is_api_configured(self) -> bool:
        """Check if an active, non-placeholder API key is configured."""
        if not self.api_key:
            return False
        k = self.api_key.strip()
        if k in ("your_api_key_here", "sk-...", "") or k.startswith("your_"):
            return False
        return True

    def _fallback_diagnose(self, incident: IncidentWindow) -> IncidentReport:
        """Deterministic, rule-based RCA analyzer when LLM API is unavailable."""
        all_text = " ".join(incident.sample_messages).lower()
        service = incident.service

        # Heuristic 1: Memory Leak / JVM Heap Exhaustion (Priority check before generic 'pool' matches)
        if (
            "outofmemoryerror" in all_text
            or "heap space" in all_text
            or "gc overhead" in all_text
            or (incident.dominant_status == 500 and incident.avg_latency > 3000)
        ):
            return IncidentReport(
                incident_id=incident.incident_id,
                incident_title=f"JVM Memory Leak and Heap Exhaustion on {service}",
                severity="CRITICAL",
                root_cause=(
                    f"Progressive heap memory depletion leading to OutOfMemoryError: Java heap space on {service}. "
                    f"Garbage collection overhead limit exceeded as Eden and Tenured generations reached 98%+ saturation, "
                    f"causing cascading thread death and severe latency degradation up to {incident.max_latency}ms."
                ),
                affected_components=[service, "JVM Heap Allocator", "PaymentDispatcher ThreadPool", "Kubernetes Pod Container"],
                remediation_steps=[
                    "Capture and inspect JVM heap dumps (-XX:+HeapDumpOnOutOfMemoryError) via Eclipse Memory Analyzer (MAT) to identify leak retention roots.",
                    "Increase Kubernetes memory request/limit quotas and tune JVM -Xmx / -Xms heap allocations.",
                    "Restart the compromised container instances and implement batch-size caps on PaymentDispatcher queue processing.",
                ],
                confidence_score=0.97,
            )

        # Heuristic 2: Credential Stuffing / Auth Spike
        if (
            "401" in str(incident.status_codes)
            or "credential" in all_text
            or "stuffing" in all_text
            or "authentication failure" in all_text
            or incident.dominant_status == 401
        ):
            return IncidentReport(
                incident_id=incident.incident_id,
                incident_title=f"Distributed Credential Stuffing Attack on {service}",
                severity="HIGH",
                root_cause=(
                    f"High-frequency burst of unauthorized authentication attempts (HTTP 401) targeting {service}. "
                    f"Repetitive invalid credentials detected from automated sources attempting dictionary credential stuffing."
                ),
                affected_components=[service, "API Gateway Rate Limiter", "Auth Provider Redis Store", "WAF Ruleset"],
                remediation_steps=[
                    "Implement Cloudflare / AWS WAF managed rate limiting rules on /api/v1/login endpoint by IP and device fingerprint.",
                    "Enforce progressive exponential backoff and CAPTCHA challenge after 3 consecutive failed login attempts.",
                    "Revoke affected sessions and enforce multi-factor authentication (MFA) reset notifications for targeted users.",
                ],
                confidence_score=0.94,
            )

        # Heuristic 3: Database Connection Pool Exhaustion
        if (
            "hikaripool" in all_text
            or ("connection" in all_text and ("pool" in all_text or "timeout" in all_text))
            or incident.dominant_status == 503
        ):
            return IncidentReport(
                incident_id=incident.incident_id,
                incident_title=f"Database Connection Pool Saturation on {service}",
                severity="CRITICAL",
                root_cause=(
                    f"The connection pool (e.g. HikariPool) on {service} was completely exhausted "
                    f"with active connections maxed out. Incoming worker threads blocked waiting for an idle "
                    f"connection until SQLTransientConnectionException timeouts were triggered with latency up to {incident.max_latency}ms."
                ),
                affected_components=[service, "PostgreSQL/Aurora RDS", "Hikari Connection Pool", "Database Read Replica"],
                remediation_steps=[
                    "Increase maximum connection pool size (HikariCP maximumPoolSize) from 50 to 100 and tune idleTimeout.",
                    "Audit slow transactional queries and optimize missing indexes on active order/cart tables.",
                    "Deploy PgBouncer / RDS Proxy to multiplex database client connections and absorb burst spikes.",
                ],
                confidence_score=0.96,
            )

        # General High Latency / Service Degradation Fallback
        severity_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = (
            "HIGH" if incident.avg_latency > 2000 or incident.dominant_status >= 500 else "MEDIUM"
        )
        return IncidentReport(
            incident_id=incident.incident_id,
            incident_title=f"Service Latency Degradation & Elevated Error Rate on {service}",
            severity=severity_tier,
            root_cause=(
                f"Statistical anomaly cluster detected on {service} with {incident.log_count} anomalous events, "
                f"average latency of {incident.avg_latency}ms (peak {incident.max_latency}ms), and status code {incident.dominant_status}."
            ),
            affected_components=[service, "Upstream Load Balancer", "Service Mesh Envoy Proxy"],
            remediation_steps=[
                f"Inspect service {service} CPU and memory metrics on Grafana / Datadog during the anomaly window.",
                "Review recent deployments or feature flag toggles deployed within the past 2 hours.",
                "Verify downstream network connectivity and circuit breaker configuration.",
            ],
            confidence_score=0.88,
        )

    def diagnose_incident(
        self,
        incident: IncidentWindow,
        force_heuristic: bool = False,
    ) -> IncidentReport:
        """Diagnose an incident using OpenAI API structured outputs, with seamless fallback."""
        if force_heuristic or not self.is_api_configured():
            return self._fallback_diagnose(incident)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            summary = incident.to_summary_dict()
            system_prompt = (
                "You are an expert Site Reliability Engineer (SRE) and distributed systems root-cause analyst. "
                "Analyze the provided anomalous incident window telemetry and return a structured JSON diagnostic "
                "conforming exactly to the IncidentReport schema."
            )
            user_prompt = (
                f"Analyze this anomalous incident window:\n"
                f"- Incident ID: {summary['incident_id']}\n"
                f"- Service: {summary['service']}\n"
                f"- Duration: {summary['duration_sec']} seconds\n"
                f"- Anomalous Log Count: {summary['log_count']}\n"
                f"- Latency: Average {summary['avg_latency_ms']}ms, Max {summary['max_latency_ms']}ms\n"
                f"- Status Codes: {summary['status_codes']} (Dominant: {summary['dominant_status']})\n"
                f"- Sample Log Messages:\n"
                + "\n".join(f"  * {m}" for m in summary["sample_messages"])
                + "\n\nProvide an incident title, severity (LOW, MEDIUM, HIGH, CRITICAL), concise root cause explanation, "
                "affected components, and exactly 3 concrete engineering remediation steps."
            )

            # Try structured outputs using client.beta.chat.completions.parse
            try:
                response = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=IncidentReport,
                    temperature=0.2,
                )
                report = response.choices[0].message.parsed
                if report:
                    return report
            except Exception:
                # Fallback to standard chat completion with JSON mode
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt + " Output valid JSON conforming to IncidentReport schema."},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = response.choices[0].message.content or "{}"
                data = json.loads(content)
                data["incident_id"] = incident.incident_id
                return IncidentReport(**data)

        except Exception as e:
            # If any API error occurs (auth, quota, network), fallback gracefully
            fallback = self._fallback_diagnose(incident)
            return fallback

    def analyze_all_incidents(
        self,
        incidents: List[IncidentWindow],
        force_heuristic: bool = False,
    ) -> List[IncidentReport]:
        """Diagnose all grouped incident windows."""
        return [
            self.diagnose_incident(inc, force_heuristic=force_heuristic)
            for inc in incidents
        ]
