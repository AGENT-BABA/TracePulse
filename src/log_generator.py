"""Synthetic Log Generator for TracePulse.

Generates realistic microservice system/API logs in standard format:
YYYY-MM-DD HH:MM:SS,mmm | LEVEL | SERVICE | ENDPOINT | LATENCY_MS | STATUS_CODE | MESSAGE

Produces normal baseline operational logs (~97%) and injects 3 distinct
anomaly clusters (~3%):
1. Database Connection Pool Exhaustion (order-api, 503, high latency)
2. Credential Stuffing / Auth Spike (auth-service, 401 burst, /api/v1/login)
3. Memory Leak / OOM Crash (payment-gateway, rising latency + 500 OutOfMemoryError)
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple


class SyntheticLogGenerator:
    """Configurable generator for synthetic microservice logs with realistic anomaly clusters."""

    SERVICES_CONFIG = {
        "auth-service": {
            "endpoints": [
                "/api/v1/auth/verify",
                "/api/v1/auth/token",
                "/api/v1/session/validate",
                "/api/v1/users/permissions",
            ],
            "messages": [
                "Session token verified successfully for user_id={uid}",
                "JWT issued for principal={principal} scope=read,write",
                "OAuth2 authorization code exchanged for token",
                "Permission check granted for role={role}",
                "Public key cache refreshed from discovery endpoint",
            ],
            "weight": 0.28,
        },
        "order-api": {
            "endpoints": [
                "/api/v1/orders/create",
                "/api/v1/orders/status",
                "/api/v1/orders/history",
                "/api/v1/cart/checkout",
            ],
            "messages": [
                "Order order_{order_id} created successfully for customer_{uid}",
                "Order status retrieved: status=SHIPPED items={item_count}",
                "Cart validated with inventory reservation id={res_id}",
                "Calculated tax and promotion discounts for cart_{order_id}",
                "Dispatched order fulfillment event to Kafka topic order.events",
            ],
            "weight": 0.25,
        },
        "payment-gateway": {
            "endpoints": [
                "/api/v1/payments/process",
                "/api/v1/payments/refund",
                "/api/v1/cards/validate",
                "/api/v1/webhooks/stripe",
            ],
            "messages": [
                "Payment transaction txn_{txn_id} settled with acquirer in {lat}ms",
                "Card tokenization succeeded for vault_id={vault_id}",
                "Webhook event payment.succeeded processed for intent_{txn_id}",
                "Pre-authorization approved for amount=${amount}.00",
                "3DS verification completed with challenge=false",
            ],
            "weight": 0.22,
        },
        "inventory-service": {
            "endpoints": [
                "/api/v1/inventory/check",
                "/api/v1/inventory/reserve",
                "/api/v1/warehouse/stock",
            ],
            "messages": [
                "Stock availability confirmed for sku={sku} count={item_count}",
                "Inventory reservation reserved 2 units of sku={sku}",
                "Warehouse sync completed for region us-east-1",
            ],
            "weight": 0.13,
        },
        "user-service": {
            "endpoints": [
                "/api/v1/users/profile",
                "/api/v1/users/settings",
                "/api/v1/users/notifications",
            ],
            "messages": [
                "User profile loaded from Redis cache key=user:{uid}",
                "Notification preferences updated for user_{uid}",
                "User metadata synced to CRM provider",
            ],
            "weight": 0.12,
        },
    }

    def __init__(self, seed: int = 42) -> None:
        """Initialize generator with fixed seed for reproducibility."""
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def _format_timestamp(dt: datetime) -> str:
        """Format timestamp as YYYY-MM-DD HH:MM:SS,mmm efficiently."""
        return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    def _generate_normal_entry(self, current_time: datetime) -> Tuple[datetime, str]:
        """Generate a single normal baseline log entry."""
        services = list(self.SERVICES_CONFIG.keys())
        weights = [self.SERVICES_CONFIG[s]["weight"] for s in services]
        service = random.choices(services, weights=weights, k=1)[0]
        cfg = self.SERVICES_CONFIG[service]

        endpoint = random.choice(cfg["endpoints"])
        raw_msg = random.choice(cfg["messages"])

        # Realistic normal latency (20ms to 180ms with normal centering around 55ms)
        latency = round(min(180.0, max(20.0, random.gauss(55.0, 24.0))), 2)

        # Baseline level and status code (97.5% 200/201, 2.5% mild WARN)
        if random.random() < 0.025:
            level = "WARN"
            status = random.choice([200, 204])
            msg = f"Slow query advisory: operation completed in {latency}ms on {service} cache"
        else:
            level = "INFO"
            status = random.choice([200, 200, 200, 201])
            msg = raw_msg.format(
                uid=random.randint(1000, 9999),
                principal=f"user_{random.randint(100, 999)}",
                role=random.choice(["customer", "merchant", "operator"]),
                order_id=random.randint(100000, 999999),
                item_count=random.randint(1, 8),
                res_id=f"res_{random.randint(1000, 9999)}",
                txn_id=random.randint(50000, 99999),
                vault_id=f"vlt_{random.randint(100, 999)}",
                lat=latency,
                amount=random.randint(15, 350),
                sku=f"SKU-{random.randint(100, 999)}",
            )

        log_line = (
            f"{self._format_timestamp(current_time)} | {level} | {service} | "
            f"{endpoint} | {latency:.2f} | {status} | {msg}"
        )
        return current_time, log_line

    def _generate_db_pool_cluster(
        self, start_time: datetime, count: int = 16
    ) -> List[Tuple[datetime, str]]:
        """Cluster 1: Database Connection Pool Exhaustion on order-api."""
        entries: List[Tuple[datetime, str]] = []
        curr = start_time
        service = "order-api"
        endpoint = "/api/v1/orders/create"

        for i in range(count):
            curr += timedelta(milliseconds=random.randint(800, 2500))
            latency = round(random.uniform(2500.0, 8000.0), 2)
            active_conn = random.randint(48, 50)
            wait_time = int(latency * 0.95)

            if i % 3 == 0:
                level = "CRITICAL"
                msg = (
                    f"Timeout waiting for connection from pool: HikariPool-1 "
                    f"(active={active_conn}, idle=0, wait_time={wait_time}ms) - thread {random.randint(10, 40)}"
                )
            elif i % 3 == 1:
                level = "ERROR"
                msg = (
                    f"Database pool exhausted: could not acquire connection within timeout={wait_time}ms "
                    f"during order creation for cart_{random.randint(1000, 9999)}"
                )
            else:
                level = "ERROR"
                msg = (
                    f"Transaction aborted: SQLTransientConnectionException: HikariPool-1 - "
                    f"Connection is not available, request timed out after {wait_time}ms"
                )

            status = 503
            log_line = (
                f"{self._format_timestamp(curr)} | {level} | {service} | "
                f"{endpoint} | {latency:.2f} | {status} | {msg}"
            )
            entries.append((curr, log_line))

        return entries

    def _generate_auth_stuffing_cluster(
        self, start_time: datetime, count: int = 18
    ) -> List[Tuple[datetime, str]]:
        """Cluster 2: Credential Stuffing / Auth Spike on auth-service."""
        entries: List[Tuple[datetime, str]] = []
        curr = start_time
        service = "auth-service"
        endpoint = "/api/v1/login"
        attacker_ip = "198.51.100.42"
        usernames = ["admin", "root", "administrator", "sysadmin", "service_account", "test_user"]

        for _ in range(count):
            curr += timedelta(milliseconds=random.randint(200, 800))
            latency = round(random.uniform(35.0, 110.0), 2)
            target_user = random.choice(usernames)
            status = 401

            r = random.random()
            if r < 0.60:
                level = "WARN"
                msg = (
                    f"Failed authentication attempt for user '{target_user}' from IP {attacker_ip}: "
                    f"Invalid credentials"
                )
            elif r < 0.85:
                level = "ERROR"
                msg = (
                    f"Rapid authentication failure burst detected: 5 consecutive 401s from IP {attacker_ip} "
                    f"targeting account '{target_user}'"
                )
            else:
                level = "CRITICAL"
                msg = (
                    f"Security anomaly: Potential credential stuffing attack from IP {attacker_ip} "
                    f"triggered IP reputation threshold alert"
                )

            log_line = (
                f"{self._format_timestamp(curr)} | {level} | {service} | "
                f"{endpoint} | {latency:.2f} | {status} | {msg}"
            )
            entries.append((curr, log_line))

        return entries

    def _generate_oom_leak_cluster(
        self, start_time: datetime, count: int = 15
    ) -> List[Tuple[datetime, str]]:
        """Cluster 3: Memory Leak / JVM OOM Crash on payment-gateway."""
        entries: List[Tuple[datetime, str]] = []
        curr = start_time
        service = "payment-gateway"
        endpoint = "/api/v1/payments/process"

        base_latencies = [750, 1200, 1900, 2800, 3900, 5200, 6800, 7900, 8400, 8900, 9200, 9500, 9700, 9850, 9900]

        for i in range(count):
            curr += timedelta(milliseconds=random.randint(600, 1800))
            idx = min(i, len(base_latencies) - 1)
            latency = round(base_latencies[idx] + random.uniform(-100, 200), 2)

            if i < 4:
                level = "WARN"
                status = 200
                msg = (
                    f"GC pause time exceeding SLO threshold: ParallelGC took {int(latency*0.4)}ms; "
                    f"heap usage at {82 + i * 3}%"
                )
            elif i < 8:
                level = "ERROR"
                status = 500
                msg = (
                    f"GC overhead limit exceeded during transaction processing for txn_{random.randint(10000, 99999)}: "
                    f"Eden and Tenured space at 98.7% saturation"
                )
            else:
                level = "CRITICAL"
                status = 500
                msg = (
                    f"java.lang.OutOfMemoryError: Java heap space at PaymentDispatcher.processBatch() "
                    f"[thread: pool-payment-worker-{random.randint(1, 16)}]"
                )

            log_line = (
                f"{self._format_timestamp(curr)} | {level} | {service} | "
                f"{endpoint} | {latency:.2f} | {status} | {msg}"
            )
            entries.append((curr, log_line))

        return entries

    def generate_logs(
        self,
        normal_count: int = 1500,
        start_time: datetime | None = None,
    ) -> List[str]:
        """Generate a complete chronologically ordered stream of normal and anomalous logs."""
        if start_time is None:
            start_time = datetime(2026, 11, 25, 9, 0, 0)

        all_entries: List[Tuple[datetime, str]] = []

        curr_time = start_time
        # For large volumes (e.g. 50,000), advance time by smaller increments so time stays compact
        step_ms = max(20, min(1200, int(1800000 / max(normal_count, 1))))

        for _ in range(normal_count):
            curr_time += timedelta(milliseconds=random.randint(step_ms // 2, step_ms * 2))
            all_entries.append(self._generate_normal_entry(curr_time))

        total_span = max(1.0, (curr_time - start_time).total_seconds())

        # Scale cluster sizes proportionally if generating high volume
        c1_count = max(16, min(50, int(normal_count * 0.01)))
        c2_count = max(18, min(60, int(normal_count * 0.012)))
        c3_count = max(15, min(50, int(normal_count * 0.01)))

        # Inject Cluster 1: Database Connection Pool Exhaustion at ~25% timestamp
        t_cluster1 = start_time + timedelta(seconds=total_span * 0.25)
        all_entries.extend(self._generate_db_pool_cluster(t_cluster1, count=c1_count))

        # Inject Cluster 2: Credential Stuffing / Auth Spike at ~55% timestamp
        t_cluster2 = start_time + timedelta(seconds=total_span * 0.55)
        all_entries.extend(self._generate_auth_stuffing_cluster(t_cluster2, count=c2_count))

        # Inject Cluster 3: Memory Leak / OOM Crash at ~80% timestamp
        t_cluster3 = start_time + timedelta(seconds=total_span * 0.80)
        all_entries.extend(self._generate_oom_leak_cluster(t_cluster3, count=c3_count))

        # Sort all entries chronologically by timestamp
        all_entries.sort(key=lambda x: x[0])

        return [line for _, line in all_entries]

    def save_to_file(self, output_path: str | Path, normal_count: int = 1500) -> str:
        """Generate logs and persist to the target file path."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = self.generate_logs(normal_count=normal_count)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(out.resolve())


def main() -> None:
    """CLI entry point to generate sample_logs.log into data/."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    data_dir = project_root / "data"
    output_file = data_dir / "sample_logs.log"

    print("Generating synthetic logs with TracePulse generator...")
    generator = SyntheticLogGenerator(seed=42)
    saved_path = generator.save_to_file(output_file, normal_count=1500)
    print(f"Successfully generated sample logs at: {saved_path}")


if __name__ == "__main__":
    main()
