#!/usr/bin/env python3

import requests
import pandas as pd
import time
import numpy as np
import os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed  # <-- added

from kubernetes import client, config

# Load kubeconfig
config.load_kube_config()

core_v1 = client.CoreV1Api()
metrics_v1 = client.CustomObjectsApi()

os.makedirs("traffic_results", exist_ok=True)


def get_pod_stats(service_name):
    namespace = "default"

    pod_list = core_v1.list_namespaced_pod(
        namespace,
        label_selector=f"serving.knative.dev/service={service_name}"
    ).items

    pod_count = len(pod_list)

    total_cpu_m = 0.0
    total_mem_mib = 0.0

    if pod_count == 0:
        return 0, 0.0, 0.0

    try:
        metrics = metrics_v1.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods"
        )

        for pod in metrics["items"]:
            if pod["metadata"]["name"] in [p.metadata.name for p in pod_list]:
                for container in pod["containers"]:
                    cpu = container["usage"]["cpu"]
                    mem = container["usage"]["memory"]

                    if cpu.endswith("n"):
                        cpu_m = int(cpu[:-1]) / 1e6
                    elif cpu.endswith("u"):
                        cpu_m = int(cpu[:-1]) / 1000
                    elif cpu.endswith("m"):
                        cpu_m = float(cpu[:-1])
                    else:
                        cpu_m = float(cpu) * 1000

                    if mem.endswith("Ki"):
                        mem_mib = float(mem[:-2]) / 1024
                    elif mem.endswith("Mi"):
                        mem_mib = float(mem[:-2])
                    elif mem.endswith("Gi"):
                        mem_mib = float(mem[:-2]) * 1024
                    else:
                        mem_mib = float(mem) / (1024 * 1024)

                    total_cpu_m += cpu_m
                    total_mem_mib += mem_mib

    except Exception:
        return pod_count, 0.0, 0.0

    return pod_count, total_cpu_m, total_mem_mib


def get_service_url(function_id, model_type):
    import subprocess
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


def generate_traffic(function_id, model_type, data_file, max_minutes=200, speedup=90):

    url, service_name = get_service_url(function_id, model_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"traffic_results/{function_id}_{model_type}_{timestamp}.log"

    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\n TRAFFIC RUN: {function_id}-{model_type}")
    log(f"URL: {url}")

    csv_path = f"traffic_results/{function_id}_{model_type}_{timestamp}.csv"

    df = pd.read_csv(data_file)
    traffic = df["y"].values

    duration = min(len(traffic), max_minutes)
    traffic = traffic[:duration]

    all_metrics = []
    prev_pods = 0

    log(f"Replay minutes: {duration}")
    log(f"Speedup: {speedup}x\n")

    for minute, req_per_min in enumerate(traffic):
        req_per_min = int(req_per_min)
        actual = min(req_per_min, 50)

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log(f"[{ts}] Minute {minute+1}/{duration}: {actual} req")

        if actual == 0:
            pods, cpu_m, mem_m = get_pod_stats(service_name)
            cold = int(prev_pods == 0 and pods > 0 or pods > prev_pods)
            prev_pods = pods

            all_metrics.append({
                "timestamp": ts,
                "function": function_id,
                "model": model_type,
                "minute": minute + 1,
                "requests_sent": 0,
                "success": 0,
                "errors": 0,
                "success_rate": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "max_latency_ms": 0,
                "pods": pods,
                "cpu_millicores": cpu_m,
                "mem_mib": mem_m,
                "cold_start": cold,
            })

            time.sleep(1 / speedup)
            continue

        latencies = []
        success = 0
        errors = 0

        delay = max((60 / actual) / (speedup * 8), 0.005)

        # send requests 
        def send_request():
            start = time.time()
            try:
                if model_type == "reactive":
                    r = requests.get(url, timeout=5)
                else:
                    payload = {"recent_data": [10] * 30, "periods": 5}
                    r = requests.post(f"{url}/predict", json=payload, timeout=5)

                latency = (time.time() - start) * 1000

                return latency, (r.status_code == 200)
            except Exception:
                return None, False

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send_request) for _ in range(actual)]

            for future in as_completed(futures):
                latency, ok = future.result()

                if latency is not None:
                    latencies.append(latency)

                if ok:
                    success += 1
                else:
                    errors += 1

        time.sleep(delay)

        pods, cpu_m, mem_m = get_pod_stats(service_name)
        cold = int(prev_pods == 0 and pods > 0 or pods > prev_pods)
        prev_pods = pods

        if len(latencies) > 0:
            avg_l = np.mean(latencies)
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            max_l = np.max(latencies)
        else:
            avg_l = p50 = p95 = p99 = max_l = 0

        all_metrics.append({
            "timestamp": ts,
            "function": function_id,
            "model": model_type,
            "minute": minute + 1,
            "requests_sent": actual,
            "success": success,
            "errors": errors,
            "success_rate": success / actual * 100 if actual > 0 else 0,
            "avg_latency_ms": avg_l,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "max_latency_ms": max_l,
            "pods": pods,
            "cpu_millicores": cpu_m,
            "mem_mib": mem_m,
            "cold_start": cold,
        })

        log(f"→ {success}/{actual} ok | pods={pods} | p95={p95:.1f} ms {'COLD' if cold else ''}")

        time.sleep(1 / speedup)

    out = pd.DataFrame(all_metrics)
    out.to_csv(csv_path, index=False)

    log("\n DONE")
    log(f"Saved results: {csv_path}")
    log(f"Total cold starts: {out['cold_start'].sum()}")
    log(f"Mean p95 latency: {out['p95_latency_ms'].mean():.1f} ms")

    log_file.close()

    return out


if __name__ == "__main__":
    import sys
    function_id = sys.argv[1]
    model_type = sys.argv[2]
    data_file = sys.argv[3]

    max_minutes = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    speedup = int(sys.argv[5]) if len(sys.argv) > 5 else 90

    generate_traffic(function_id, model_type, data_file, max_minutes, speedup)
