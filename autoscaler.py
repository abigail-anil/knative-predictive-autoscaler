import requests
import time
import random
import subprocess

ML_API_URL = "http://func-ml-autoscaler.default.127.0.0.1.sslip.io/predict"
FUNC_IDS = ["110","126","235"]  
MODEL_TYPE = "hybrid"  # prophet/lstm/reactive
KNATIVE_SERVICE_PREFIX = "func-"  # service names: func-110
SLEEP_INTERVAL = 15  # seconds between scaling checks

def get_predicted_replicas(func_id):
    traffic = [random.randint(1,50) for _ in range(10)]  # simulate traffic
    resp = requests.post(ML_API_URL, json={"func_id": func_id, "model_type": MODEL_TYPE, "recent_traffic": traffic})
    return resp.json().get("predicted_replicas", 1)

def scale_knative(func_id, replicas):
    service_name = f"{KNATIVE_SERVICE_PREFIX}{func_id}-hybrid"
    cmd = f"kubectl scale ksvc {service_name} --replicas={replicas}"
    subprocess.run(cmd, shell=True)
    print(f"Scaled {service_name} to {replicas} replicas")

if __name__ == "__main__":
    while True:
        for fid in FUNC_IDS:
            try:
                pred = get_predicted_replicas(fid)
                scale_knative(fid, pred)
            except Exception as e:
                print(f"Error scaling {fid}: {e}")
        time.sleep(SLEEP_INTERVAL)
