#!/usr/bin/env python3
"""
Traffic generator + observability collector for Knative on AKS.
Captures latency, success rate, pod scaling, CPU/memory, and cold-start events.
"""

import requests
import pandas as pd
import time
import subprocess
import numpy as np
import os
from datetime import datetime, timezone

# Setup
os.makedirs("traffic_results", exist_ok=True)


# Helper functions
def get_service_url(function_id, model_type):
    """Resolve Kourier external IP and form Knative service URL."""
    result = subprocess.run(
        [
            "kubectl", "get", "svc", "kourier",
            "-n", "kourier-system",
            "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
        ],
        capture_output=True, text=True
    )
    external_ip = result.stdout.strip()
    service_name = f"{function_id.replace('_', '-')}-{model_type}"
    return f"http://{service_name}.default.{external_ip}.sslip.io", service_name


def get_pod_stats(service_name):
    """Return current pod count + aggregated CPU/memory usage."""
    try:
        pods = subprocess.run(
            [
                "kubectl", "get", "pods", "-l",
                f"serving.knative.dev/service={service_name}",
                "-o", "jsonpath={.items[*].metadata.name}"
            ],
            capture_output=True, text=True
        ).stdout.strip().split()

        pod_count = len([p for p in pods if p])
        cpu_total, mem_total = 0.0, 0.0

        if pod_count > 0:
            top_out = subprocess.run(
                [
                    "kubectl", "top", "pods", "--no-headers", "-l",
                    f"serving.knative.dev/service={service_name}"
                ],
                capture_output=True, text=True
            ).stdout.strip().splitlines()

            for line in top_out:
                parts = line.split()
                if len(parts) >= 3:
                    cpu = float(parts[1].replace("m", ""))
                    mem = float(parts[2].replace("Mi", ""))
                    cpu_total += cpu
                    mem_total += mem

        return pod_count, cpu_total, mem_total
    except Exception:
        return 0, 0.0, 0.0


# Main traffic generation

def generate_traffic(function_id, model_type, data_file, duration=10, speedup=60):
    """Generate synthetic or replayed traffic and log detailed metrics."""
    url, service_name = get_service_url(function_id, model_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"traffic_results/{function_id}_{model_type}_{timestamp}.log"
    csv_path = f"traffic_results/{function_id}_{model_type}_{timestamp}.csv"

    with open(log_path, "w", encoding="utf-8") as log_file:
        def log(msg):
            print(msg)
            log_file.write(msg + "\n")


        log(f"TRAFFIC RUN: {function_id}-{model_type}")
        log(f"Target URL: {url}")


        df = pd.read_csv(data_file)
        duration = min(duration, len(df))
        traffic = df["y"].values[:duration]

        all_metrics = []
        prev_pod_count = 0

        for minute, requests_per_min in enumerate(traffic):
            requests_per_min = int(requests_per_min)
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            log(f"[{ts}] Minute {minute+1}/{duration}: {requests_per_min} req",)

            if requests_per_min == 0:
                time.sleep(60 / speedup)
                continue

            actual = min(requests_per_min, 50)
            #delay = (60 / actual) / speedup
            delay = max((60 / actual) / (speedup * 5), 0.01)

            latencies, success, errors = [], 0, 0

            # ---- send requests ----
            for _ in range(actual):
                start = time.time()
                try:
                    if model_type == "reactive":
                        r = requests.get(url, timeout=5)
                    else:
                        payload = {"recent_data": [10] * 30, "periods": 5}
                        r = requests.post(f"{url}/predict", json=payload, timeout=5)

                    latency = (time.time() - start) * 1000
                    latencies.append(latency)
                    if r.status_code == 200:
                        success += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1
                time.sleep(delay)

            # ---- collect pod stats ----
            pod_count, cpu_m, mem_m = get_pod_stats(service_name)

            # ---- detect cold start ----
            cold_start_flag = int(prev_pod_count == 0 and pod_count > 0 or pod_count > prev_pod_count)
            prev_pod_count = pod_count

            metrics = {
                "timestamp": ts,
                "function": function_id,
                "model": model_type,
                "minute": minute + 1,
                "requests_sent": actual,
                "success": success,
                "errors": errors,
                "success_rate": (success / actual * 100) if actual > 0 else 0,
                "avg_latency_ms": np.mean(latencies) if latencies else 0,
                "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
                "max_latency_ms": np.max(latencies) if latencies else 0,
                "pods": pod_count,
                "cpu_millicores": cpu_m,
                "mem_mib": mem_m,
                "cold_start": cold_start_flag,
            }
            all_metrics.append(metrics)

            cold = "cold" if cold_start_flag else ""
            log(
                f"→ {success}/{actual} ok | pods={pod_count} | "
                f"avg={metrics['avg_latency_ms']:.1f}ms | "
                f"p95={metrics['p95_latency_ms']:.1f}ms {cold}"
            )

            time.sleep(1 / speedup)

        # ---- save results ----
        df_out = pd.DataFrame(all_metrics)
        df_out.to_csv(csv_path, index=False)

        log("\n" )
        log(f"Run complete. Detailed metrics saved to {csv_path}")
        log(f"Mean success: {df_out['success_rate'].mean():.1f}%")
        log(f"Avg p95 latency: {df_out['p95_latency_ms'].mean():.1f} ms")
        log(f"Cold starts detected: {df_out['cold_start'].sum()}")


    return df_out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        sys.exit(1)

    function_id = sys.argv[1]
    model_type = sys.argv[2]
    data_file = sys.argv[3]
    duration = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    speedup = int(sys.argv[5]) if len(sys.argv) > 5 else 60

    generate_traffic(function_id, model_type, data_file, duration, speedup)
