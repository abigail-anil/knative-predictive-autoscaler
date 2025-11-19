#!/usr/bin/env python3
"""
Predictive autoscaler for Knative on AKS 
"""

import requests
import subprocess
import json
import time
import logging
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AKSAutoscaler:
    def __init__(self, function_id, model_type, use_real_traffic=False, traffic_csv=None):
        self.function_id = function_id
        self.model_type = model_type
        self.service_name = f"{function_id.replace('_','-')}-{model_type}"

        self.use_real_traffic = use_real_traffic
        self.traffic_data = None
        self.traffic_index = 0

        # TUNED PARAMETERS FOR REALISTIC AUTOSCALING
        self.max_pods = 5
        self.min_pods = 0  # Allow scale-to-zero
        

        self.requests_per_pod = 30  # Each pod handles 30 req/min
        
        self.prediction_horizon = 5  # Predict 5 minutes ahead
        self.lookback = 30
        
        self.scale_up_threshold = 0.8    # Scale up if load > 80% capacity
        self.scale_down_threshold = 0.5  # Scale down if load < 50% capacity
        self.cooldown_period = 30        # Seconds between scale operations

        self.last_scale_time = 0

        # Load traffic CSV
        if traffic_csv and os.path.exists(traffic_csv):
            self.traffic_data = pd.read_csv(traffic_csv)
            logger.info(f"Loaded {len(self.traffic_data)} traffic datapoints")
            
            # Show traffic stats to understand patterns
            traffic_stats = self.traffic_data['y'].describe()
            logger.info(f"Traffic stats: min={traffic_stats['min']:.0f}, "
                       f"mean={traffic_stats['mean']:.0f}, "
                       f"max={traffic_stats['max']:.0f}")

        # Seed initial history
        self.history = deque(maxlen=60)
        if self.traffic_data is not None:
            for i in range(min(self.lookback, len(self.traffic_data))):
                self.history.append(float(self.traffic_data['y'].iloc[i]))
        else:
            for _ in range(self.lookback):
                self.history.append(10.0)

        # External IP
        self.external_ip = self.get_external_ip()
        self.service_url = f"http://{self.service_name}.default.{self.external_ip}.sslip.io"

        logger.info(f"Kourier IP: {self.external_ip}")
        logger.info(f"Service URL: {self.service_url}")

        self.test_service_connection()

    def get_external_ip(self):
        result = subprocess.run(
            "kubectl get svc kourier -n kourier-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'",
            capture_output=True, text=True, shell=True
        )
        return result.stdout.strip().strip("'")

    def test_service_connection(self):
        try:
            logger.info("Testing service connectivity...")
            r = requests.get(f"{self.service_url}/health", timeout=120)
            logger.info(f"Service reachable: {r.json()}")
        except Exception as e:
            logger.error(f" Service NOT reachable: {e}")

    def get_current_metrics(self):
        """Get current pod count and request rate"""
        result = subprocess.run(
            ["kubectl","get","pods","-l",f"serving.knative.dev/service={self.service_name}","-o","json"],
            capture_output=True, text=True
        )
        pods_json = json.loads(result.stdout)
        
        # Count running AND pending pods (for cold start awareness)
        running_pods = sum(1 for p in pods_json.get("items",[]) if p["status"]["phase"]=="Running")
        pending_pods = sum(1 for p in pods_json.get("items",[]) if p["status"]["phase"]=="Pending")
        total_pods = running_pods + pending_pods

        # Get request rate from traffic data or simulation
        if self.use_real_traffic and self.traffic_data is not None:
            if self.traffic_index >= len(self.traffic_data):
                self.traffic_index = 0
            request_rate = float(self.traffic_data['y'].iloc[self.traffic_index])
            self.traffic_index += 1
        else:
            # Simulate varying traffic
            request_rate = max(5, np.random.normal(40, 15))

        return {
            "pod_count": total_pods,
            "running_pods": running_pods,
            "pending_pods": pending_pods,
            "request_rate": request_rate
        }

    def predict_future_load(self):
        """Get predictions from ML model"""
        recent_data = list(self.history)[-self.lookback:]
        
        try:
            r = requests.post(
                f"{self.service_url}/predict",
                json={"periods": self.prediction_horizon, "recent_data": recent_data},
                timeout=30
            )
            
            if r.status_code != 200:
                logger.error(f"Prediction failed ({r.status_code}): {r.text}")
                return None

            preds = r.json()["predictions"]
            
            # Validate predictions
            if all(p == 0 for p in preds):
                logger.warning("  All predictions are zero - using current traffic as fallback")
                return [self.history[-1]] * self.prediction_horizon
            
            logger.info(f"{self.model_type.upper()} predictions: {[f'{p:.1f}' for p in preds[:3]]}...")
            return preds

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def calculate_desired_pods(self, predictions, current_pods):
        """Calculate desired pods with smart logic"""
        if not predictions:
            logger.warning("No predictions - maintaining current scale")
            return current_pods

        # Use max predicted load (pessimistic for safety)
        peak_load = max(predictions)
        
        # Fallback if prediction is zero
        if peak_load <= 0:
            peak_load = self.history[-1] if self.history else 10.0
            logger.warning(f"Using fallback load: {peak_load:.1f}")

        # Calculate raw desired pods
        raw_desired = peak_load / self.requests_per_pod
        
        # Apply scaling thresholds to prevent flapping
        current_capacity = current_pods * self.requests_per_pod
        utilization = peak_load / max(current_capacity, 1)
        
        if utilization > self.scale_up_threshold:
            # Need to scale up
            desired = int(np.ceil(raw_desired))
            action = "SCALE UP"
        elif utilization < self.scale_down_threshold:
            # Can scale down
            desired = max(self.min_pods, int(np.floor(raw_desired)))
            action = "SCALE DOWN"
        else:
            # Stay at current level (hysteresis)
            desired = current_pods
            action = "MAINTAIN"
        
        # Enforce limits
        desired = max(self.min_pods, min(desired, self.max_pods))
        
        logger.info(f" Peak load: {peak_load:.1f} req/min | "
                   f"Current capacity: {current_capacity:.0f} req/min ({current_pods} pods)")
        logger.info(f" Utilization: {utilization*100:.1f}% | "
                   f"Action: {action} | "
                   f"Desired: {desired} pods")
        
        return desired

    def scale_service(self, desired):
        """Scale the service by updating minScale"""
        # Check cooldown
        now = time.time()
        if now - self.last_scale_time < self.cooldown_period:
            remaining = self.cooldown_period - (now - self.last_scale_time)
            logger.info(f"Cooldown active ({remaining:.0f}s remaining)")
            return False
        
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/min-scale": str(desired)
                        }
                    }
                }
            }
        }

        result = subprocess.run(
            ["kubectl", "patch", "ksvc", self.service_name, 
             "--type", "merge", "-p", json.dumps(patch_body)],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            logger.info(f" Scaled to {desired} pods (minScale updated)")
            self.last_scale_time = now
            return True
        else:
            logger.error(f"Scaling failed: {result.stderr}")
            return False

    def run(self, interval=30):
        """Main autoscaling loop"""
        logger.info(f" PREDICTIVE AUTOSCALER STARTED")
        logger.info(f"   Service: {self.service_name}")
        logger.info(f"   Model: {self.model_type.upper()}")
        logger.info(f"   Traffic: {'Real CSV Data' if self.use_real_traffic else 'Simulated'}")
        logger.info(f"   Capacity: {self.requests_per_pod} req/min per pod")
        logger.info(f"   Range: {self.min_pods}-{self.max_pods} pods")

        iteration = 0

        try:
            while True:
                iteration += 1
                logger.info(f"\n")
                logger.info(f"[Iteration {iteration}] {datetime.now().strftime('%H:%M:%S')}")

                # Get current state
                metrics = self.get_current_metrics()
                pod_count = metrics["pod_count"]
                running_pods = metrics["running_pods"]
                pending_pods = metrics["pending_pods"]
                req_rate = metrics["request_rate"]

                # Update history
                self.history.append(req_rate)
                
                # Log current state
                pod_status = f"{running_pods} running"
                if pending_pods > 0:
                    pod_status += f" + {pending_pods} pending"
                
                logger.info(f" Current state: {pod_status} | {req_rate:.1f} req/min")

                # Get predictions
                preds = self.predict_future_load()
                
                # Calculate desired pods
                desired = self.calculate_desired_pods(preds, pod_count)

                # Scale if needed
                if desired != pod_count:
                    logger.info(f"Scaling: {pod_count} → {desired} pods")
                    self.scale_service(desired)
                else:
                    logger.info(f" No scaling needed (already at {pod_count} pods)")

                # Wait for next iteration
                logger.info(f"Sleeping for {interval}s...")
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\n")
            logger.info(" Autoscaler stopped by user")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Predictive autoscaler for Knative',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--function-id", required=True, help="Function ID (e.g., func_235)")
    parser.add_argument("--model-type", required=True, choices=["prophet","lstm","hybrid"],
                       help="ML model type")
    parser.add_argument("--interval", type=int, default=30, 
                       help="Polling interval in seconds")
    parser.add_argument("--use-real-traffic", action="store_true",
                       help="Use real traffic patterns from CSV")
    parser.add_argument("--traffic-csv", help="Path to traffic CSV file")
    parser.add_argument("--requests-per-pod", type=int, default=30,
                       help="Request capacity per pod (req/min)")

    args = parser.parse_args()

    autoscaler = AKSAutoscaler(
        args.function_id,
        args.model_type,
        use_real_traffic=args.use_real_traffic,
        traffic_csv=args.traffic_csv
    )
    
    # Override capacity if specified
    if args.requests_per_pod:
        autoscaler.requests_per_pod = args.requests_per_pod
        logger.info(f"Pod capacity set to: {args.requests_per_pod} req/min")
    
    autoscaler.run(args.interval)