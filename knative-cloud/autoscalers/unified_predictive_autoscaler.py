#!/usr/bin/env python3
"""
Predictive Autoscaler with Real Traffic Metrics
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
        self.service_name = f"{function_id.replace('_', '-')}-{model_type}"
        self.use_real_traffic = use_real_traffic
        self.traffic_data = None
        self.traffic_index = 0
        
        # Load real traffic data if provided
        if traffic_csv and os.path.exists(traffic_csv):
            self.traffic_data = pd.read_csv(traffic_csv)
            logger.info(f"Loaded traffic data: {len(self.traffic_data)} data points")
        
        # Get external IP
        self.external_ip = self.get_external_ip()
        self.service_url = f"http://{self.service_name}.default.{self.external_ip}.sslip.io"
        logger.info(f"Kourier IP: {self.external_ip}")
        logger.info(f"Service URL: {self.service_url}")
        
        # Test connectivity
        self.test_service_connection()
        
        # Parameters
        self.prediction_horizon = 5
        self.requests_per_pod = 50
        self.min_pods = 1
        self.max_pods = 10
        self.lookback = 30
        
        # History - initialize with realistic values
        self.history = deque(maxlen=60)
        if self.traffic_data is not None:
            # Seed with actual data
            for i in range(min(self.lookback, len(self.traffic_data))):
                self.history.append(float(self.traffic_data['y'].iloc[i]))
        else:
            for _ in range(self.lookback):
                self.history.append(10.0)
    
    def get_external_ip(self):
        """Get Kourier external IP"""
        try:
            result = subprocess.run(
                "kubectl get svc kourier -n kourier-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'",
                capture_output=True, text=True, shell=True
            )
            return result.stdout.strip().strip("'")
        except Exception as e:
            logger.error(f"Error getting external IP: {e}")
            return ""
    
    def test_service_connection(self):
        """Test if the service is reachable"""
        try:
            logger.info("Testing service connectivity...")
            response = requests.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"Service is reachable: {response.json()}")
                return True
            else:
                logger.warning(f"Service returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            return False
    
    def get_current_metrics(self):
        """Get current state with optional real traffic data"""
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-l',
             f'serving.knative.dev/service={self.service_name}',
             '-o', 'json'],
            capture_output=True, text=True
        )
        
        pods = json.loads(result.stdout)
        current_pods = sum(1 for p in pods['items'] if p['status']['phase'] == 'Running')
        
        # Get request rate
        if self.use_real_traffic and self.traffic_data is not None:
            # Use actual traffic pattern
            if self.traffic_index < len(self.traffic_data):
                request_rate = float(self.traffic_data['y'].iloc[self.traffic_index])
                self.traffic_index += 1
            else:
                # Loop back to start
                self.traffic_index = 0
                request_rate = float(self.traffic_data['y'].iloc[self.traffic_index])
        else:
            # Simulate request rate
            request_rate = max(5, np.random.normal(40, 15))
        
        return {'pod_count': current_pods, 'request_rate': request_rate}
    
    def predict_future_load(self):
        """Get predictions from model"""
        try:
            recent_data = list(self.history)[-self.lookback:]
            
            logger.debug(f"Sending data: min={min(recent_data):.1f}, max={max(recent_data):.1f}, mean={np.mean(recent_data):.1f}")
            
            response = requests.post(
                f"{self.service_url}/predict",
                json={'periods': self.prediction_horizon, 'recent_data': recent_data},
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                
                # Check if predictions are all zero
                if all(p == 0 for p in predictions):
                    logger.warning("⚠ Model returned all-zero predictions - check model training!")
                
                logger.info(f"{self.model_type.upper()} predictions: {[f'{p:.1f}' for p in predictions[:3]]}...")
                return predictions
            else:
                logger.error(f"Prediction failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
        
        return None
    
    def calculate_desired_pods(self, predictions):
        """Calculate desired pods"""
        if not predictions:
            return self.min_pods
        
        max_load = max(predictions)
        
        # Handle zero predictions
        if max_load <= 0:
            logger.warning("Zero prediction - using current traffic as baseline")
            max_load = self.history[-1] if self.history else 10.0
        
        required = int(np.ceil(max_load / self.requests_per_pod))
        desired = max(self.min_pods, min(required, self.max_pods))
        
        logger.info(f"Max predicted: {max_load:.1f} req/min → Desired: {desired} pods")
        return desired
    
    def scale_service(self, desired_pods):
        """Scale service"""
        cmd = [
            'kubectl', 'patch', 'ksvc', self.service_name,
            '--type', 'json',
            '-p', f'[{{"op":"replace","path":"/spec/template/metadata/annotations/autoscaling.knative.dev~1minScale","value":"{desired_pods}"}}]'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Scaled to {desired_pods} pods")
            return True
        else:
            logger.error(f"Scaling failed: {result.stderr}")
            return False
    
    def run(self, interval=30):
        """Main loop"""
        logger.info("=" * 60)
        logger.info(f"AKS PREDICTIVE AUTOSCALER: {self.service_name}")
        logger.info(f"Model: {self.model_type.upper()}")
        logger.info(f"Traffic Source: {'Real CSV Data' if self.use_real_traffic else 'Simulated'}")
        logger.info(f"Interval: {interval}s")
        logger.info("=" * 60)
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"\n[Iteration {iteration}] {datetime.now().strftime('%H:%M:%S')}")
                
                metrics = self.get_current_metrics()
                self.history.append(metrics['request_rate'])
                
                logger.info(f"Current: {metrics['pod_count']} pods, {metrics['request_rate']:.1f} req/min")
                
                predictions = self.predict_future_load()
                
                if predictions:
                    desired = self.calculate_desired_pods(predictions)
                    if desired != metrics['pod_count']:
                        logger.info(f"Scaling: {metrics['pod_count']} → {desired}")
                        self.scale_service(desired)
                    else:
                        logger.info(f"No scaling needed (already at {desired} pods)")
                else:
                    logger.warning("Using fallback: maintaining current scale")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("\n" + "=" * 60)
                logger.info("Autoscaler stopped by user")
                logger.info("=" * 60)
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(interval)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictive autoscaler for AKS + Knative')
    parser.add_argument('--function-id', required=True, help='Function ID (e.g., func_235)')
    parser.add_argument('--model-type', required=True, choices=['prophet', 'lstm', 'hybrid'],
                       help='Model type')
    parser.add_argument('--interval', type=int, default=30, help='Polling interval in seconds')
    parser.add_argument('--use-real-traffic', action='store_true', 
                       help='Use real traffic data from CSV')
    parser.add_argument('--traffic-csv', type=str, 
                       help='Path to traffic CSV file (e.g., data/timeseries_func_235.csv)')
    
    args = parser.parse_args()
    
    autoscaler = AKSAutoscaler(
        args.function_id, 
        args.model_type,
        use_real_traffic=args.use_real_traffic,
        traffic_csv=args.traffic_csv
    )
    autoscaler.run(args.interval)