#!/usr/bin/env python3
"""
DIAGNOSTIC Autoscaler - Shows exactly what's happening
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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiagnosticAutoscaler:
    def __init__(self, function_id, model_type, traffic_csv=None):
        self.function_id = function_id
        self.model_type = model_type
        self.service_name = f"{function_id.replace('_','-')}-{model_type}"
        
        # AGGRESSIVE SETTINGS FOR DEMONSTRATION
        self.max_pods = 3
        self.min_pods = 1
        self.requests_per_pod = 20  # Very low = aggressive scaling
        self.prediction_horizon = 5
        self.lookback = 30
        
        # Load traffic
        self.traffic_data = None
        self.traffic_index = 0
        if traffic_csv and os.path.exists(traffic_csv):
            self.traffic_data = pd.read_csv(traffic_csv)
            logger.info(f"Loaded {len(self.traffic_data)} traffic data points")
        
        # Initialize history
        self.history = deque(maxlen=60)
        if self.traffic_data is not None:
            for i in range(min(self.lookback, len(self.traffic_data))):
                self.history.append(float(self.traffic_data['y'].iloc[i]))
        else:
            for _ in range(self.lookback):
                self.history.append(10.0)
        
        # Get service URL
        self.external_ip = self.get_external_ip()
        self.service_url = f"http://{self.service_name}.default.{self.external_ip}.sslip.io"
        
        logger.info(f"="*80)
        logger.info(f"DIAGNOSTIC AUTOSCALER")
        logger.info(f"Service: {self.service_name}")
        logger.info(f"URL: {self.service_url}")
        logger.info(f"Settings: {self.requests_per_pod} req/pod, max {self.max_pods} pods")
        logger.info(f"="*80)
        
        # Check current Knative settings
        self.check_knative_config()

    def get_external_ip(self):
        result = subprocess.run(
            "kubectl get svc kourier -n kourier-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'",
            capture_output=True, text=True, shell=True
        )
        return result.stdout.strip().strip("'")
    
    def check_knative_config(self):
        """Check current Knative autoscaling settings"""
        logger.info("\n" + "="*80)
        logger.info("CHECKING KNATIVE CONFIGURATION")
        logger.info("="*80)
        
        result = subprocess.run(
            ['kubectl', 'get', 'ksvc', self.service_name, '-o', 'json'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to get service config: {result.stderr}")
            return
        
        config = json.loads(result.stdout)
        annotations = config['spec']['template']['metadata'].get('annotations', {})
        
        min_scale = annotations.get('autoscaling.knative.dev/min-scale', 'NOT SET')
        max_scale = annotations.get('autoscaling.knative.dev/max-scale', 'NOT SET')
        target = annotations.get('autoscaling.knative.dev/target', 'NOT SET')
        metric = annotations.get('autoscaling.knative.dev/metric', 'NOT SET')
        
        logger.info(f"Current Knative Settings:")
        logger.info(f"  min-scale: {min_scale}")
        logger.info(f"  max-scale: {max_scale}")
        logger.info(f"  target: {target}")
        logger.info(f"  metric: {metric}")
        
        # Warnings
        if min_scale == max_scale and min_scale != 'NOT SET':
            logger.warning(f"⚠️  min-scale == max-scale ({min_scale}) - scaling is DISABLED!")
        
        if target != 'NOT SET' and int(target) < 50:
            logger.warning(f"⚠️  target is very low ({target}) - Knative may interfere")
        
        logger.info("="*80 + "\n")

    def get_current_metrics(self):
        """Get current state with detailed pod info"""
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-l', 
             f'serving.knative.dev/service={self.service_name}', '-o', 'json'],
            capture_output=True, text=True
        )
        
        pods_json = json.loads(result.stdout)
        pods = pods_json.get('items', [])
        
        running = sum(1 for p in pods if p['status']['phase'] == 'Running')
        pending = sum(1 for p in pods if p['status']['phase'] == 'Pending')
        
        logger.debug(f"Pod status: {running} running, {pending} pending")
        
        # Get request rate
        if self.traffic_data is not None:
            if self.traffic_index >= len(self.traffic_data):
                self.traffic_index = 0
            request_rate = float(self.traffic_data['y'].iloc[self.traffic_index])
            self.traffic_index += 1
        else:
            request_rate = max(5, np.random.normal(40, 15))
        
        return {
            'pod_count': running + pending,
            'running': running,
            'pending': pending,
            'request_rate': request_rate
        }

    def predict_future_load(self):
        """Get predictions with detailed logging"""
        recent_data = list(self.history)[-self.lookback:]
        
        logger.debug(f"Recent data for prediction: min={min(recent_data):.1f}, "
                    f"max={max(recent_data):.1f}, mean={np.mean(recent_data):.1f}")
        
        try:
            logger.debug(f"Calling prediction API: {self.service_url}/predict")
            
            r = requests.post(
                f"{self.service_url}/predict",
                json={"periods": self.prediction_horizon, "recent_data": recent_data},
                timeout=10
            )
            
            if r.status_code != 200:
                logger.error(f"❌ Prediction failed: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return None
            
            result = r.json()
            preds = result["predictions"]
            
            logger.info(f"✅ Predictions received: {[f'{p:.1f}' for p in preds]}")
            
            # Check for issues
            if all(p == 0 for p in preds):
                logger.warning("⚠️  All predictions are ZERO")
            elif all(p < 1 for p in preds):
                logger.warning(f"⚠️  All predictions are very low (max={max(preds):.2f})")
            elif any(p < 0 for p in preds):
                logger.error(f"❌ NEGATIVE predictions detected: {preds}")
            
            return preds
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None

    def calculate_desired_pods(self, predictions, current_pods):
        """Calculate with detailed reasoning"""
        logger.info(f"\n{'─'*80}")
        logger.info(f"SCALING DECISION")
        logger.info(f"{'─'*80}")
        
        if not predictions:
            logger.warning("No predictions - keeping current: {current_pods}")
            return current_pods
        
        peak = max(predictions)
        logger.info(f"Peak predicted load: {peak:.1f} req/min")
        
        if peak <= 0:
            peak = self.history[-1] if self.history else 10.0
            logger.warning(f"Using fallback load: {peak:.1f}")
        
        # Calculate
        current_capacity = current_pods * self.requests_per_pod
        required_capacity = peak
        utilization = peak / max(current_capacity, 1)
        
        raw_desired = peak / self.requests_per_pod
        desired = int(np.ceil(raw_desired))
        desired = max(self.min_pods, min(desired, self.max_pods))
        
        logger.info(f"Current: {current_pods} pods × {self.requests_per_pod} = {current_capacity:.0f} req/min capacity")
        logger.info(f"Required: {required_capacity:.1f} req/min")
        logger.info(f"Utilization: {utilization*100:.1f}%")
        logger.info(f"Calculated: {raw_desired:.2f} pods → Rounded to {desired}")
        logger.info(f"After limits [{self.min_pods}-{self.max_pods}]: {desired} pods")
        
        if desired > current_pods:
            logger.info(f"🔼 DECISION: SCALE UP from {current_pods} to {desired}")
        elif desired < current_pods:
            logger.info(f"🔽 DECISION: SCALE DOWN from {current_pods} to {desired}")
        else:
            logger.info(f"⏸️  DECISION: NO CHANGE (stay at {desired})")
        
        logger.info(f"{'─'*80}\n")
        
        return desired

    def scale_service(self, desired):
        """Scale with detailed output"""
        logger.info(f"\n{'='*80}")
        logger.info(f"EXECUTING SCALE OPERATION TO {desired} PODS")
        logger.info(f"{'='*80}")
        
        # Show the patch we're applying
        patch = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/min-scale": str(desired),
                            "autoscaling.knative.dev/max-scale": str(max(desired, self.max_pods))
                        }
                    }
                }
            }
        }
        
        logger.info(f"Patch JSON: {json.dumps(patch, indent=2)}")
        
        # Execute kubectl patch
        result = subprocess.run(
            ['kubectl', 'patch', 'ksvc', self.service_name, 
             '--type', 'merge', '-p', json.dumps(patch)],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✅ kubectl patch SUCCESS")
            logger.info(f"Output: {result.stdout}")
            
            # Verify the change
            time.sleep(2)
            verify = subprocess.run(
                ['kubectl', 'get', 'ksvc', self.service_name, 
                 '-o', 'jsonpath={.spec.template.metadata.annotations}'],
                capture_output=True, text=True
            )
            logger.info(f"Verified annotations: {verify.stdout}")
            
            return True
        else:
            logger.error(f"❌ kubectl patch FAILED")
            logger.error(f"Error: {result.stderr}")
            return False

    def run(self, interval=30, max_iterations=None):
        """Main loop with diagnostics"""
        iteration = 0
        
        while True:
            iteration += 1
            
            if max_iterations and iteration > max_iterations:
                logger.info(f"Reached max iterations ({max_iterations})")
                break
            
            logger.info(f"\n\n{'='*80}")
            logger.info(f"ITERATION {iteration} - {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Get metrics
            metrics = self.get_current_metrics()
            logger.info(f"📊 Current: {metrics['running']} running, "
                       f"{metrics['pending']} pending, "
                       f"{metrics['request_rate']:.1f} req/min")
            
            # Update history
            self.history.append(metrics['request_rate'])
            
            # Predict
            preds = self.predict_future_load()
            
            # Decide
            desired = self.calculate_desired_pods(preds, metrics['pod_count'])
            
            # Scale
            if desired != metrics['pod_count']:
                success = self.scale_service(desired)
                if not success:
                    logger.error("Scale operation failed - check kubectl permissions")
            else:
                logger.info("No scaling needed")
            
            # Wait
            logger.info(f"\n⏰ Sleeping {interval}s until next iteration...\n")
            time.sleep(interval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnostic autoscaler')
    parser.add_argument("--function-id", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--traffic-csv", required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, help="Stop after N iterations")
    
    args = parser.parse_args()
    
    autoscaler = DiagnosticAutoscaler(
        args.function_id,
        args.model_type,
        args.traffic_csv
    )
    
    try:
        autoscaler.run(args.interval, args.max_iterations)
    except KeyboardInterrupt:
        logger.info("\n\nStopped by user")