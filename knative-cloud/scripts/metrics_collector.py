#!/usr/bin/env python3
"""
Metrics Collector for Knative on AKS
Collects CPU, memory, pod count, and timestamps for each function-model service.
"""

import csv
import subprocess
import json
import time
from datetime import datetime

def get_pod_metrics(label_selector):
    """Fetch CPU, memory, and status for pods matching the label."""
    pods_info = []
    
    # Get pod names and phases
    pod_data = subprocess.run(
        ["kubectl", "get", "pods", "-l", label_selector, "-o", "json"],
        capture_output=True, text=True
    )
    pods = json.loads(pod_data.stdout).get("items", [])
    
    for p in pods:
        name = p["metadata"]["name"]
        phase = p["status"]["phase"]

        # Get resource usage
        top = subprocess.run(
            ["kubectl", "top", "pod", name, "--no-headers"],
            capture_output=True, text=True
        )
        if top.returncode == 0:
            parts = top.stdout.split()
            cpu = parts[1] if len(parts) > 1 else "0m"
            mem = parts[2] if len(parts) > 2 else "0Mi"
        else:
            cpu, mem = "0m", "0Mi"

        pods_info.append({"name": name, "cpu": cpu, "mem": mem, "phase": phase})
    return pods_info


def collect_metrics(function_id, model_type, duration=600, interval=10):
    """Collect metrics periodically."""
    service_label = f"serving.knative.dev/service={function_id.replace('_','-')}-{model_type}"
    filename = f"metrics_{function_id}_{model_type}.csv"
    
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["timestamp", "pod_name", "phase", "cpu", "memory"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        start = time.time()
        while time.time() - start < duration:
            timestamp = datetime.now().isoformat()
            pods = get_pod_metrics(service_label)
            
            for p in pods:
                writer.writerow({
                    "timestamp": timestamp,
                    "pod_name": p["name"],
                    "phase": p["phase"],
                    "cpu": p["cpu"],
                    "memory": p["mem"]
                })
            csvfile.flush()
            print(f"[{timestamp}] Collected {len(pods)} pod(s)")
            time.sleep(interval)
    
    print(f"Metrics saved to {filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-id", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    collect_metrics(args.function_id, args.model_type, args.duration, args.interval)
