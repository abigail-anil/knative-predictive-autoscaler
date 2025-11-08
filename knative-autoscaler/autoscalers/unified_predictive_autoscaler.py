#!/usr/bin/env python3
"""
Unified Predictive Autoscaler - Using ClusterIP
"""
import requests
import subprocess
import json
import time
import logging
from datetime import datetime
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictiveAutoscaler:
    def __init__(self, function_id, model_type):
        self.function_id = function_id
        self.model_type = model_type
        self.service_name = f"{function_id.replace('_', '-')}-{model_type}"
        
        # Autoscaling parameters
        self.prediction_horizon = 5
        self.requests_per_pod = 50
        self.min_pods = 0
        self.max_pods = 10
        self.lookback = 30
        
        # Historical data
        self.history = deque(maxlen=60)
        for _ in range(self.lookback):
            self.history.append(10.0)
        
        # Get service URL
        self.api_url = self.get_service_url()
    
    def get_service_url(self):
        """Get the internal Kubernetes service URL"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'ksvc', self.service_name, '-o', 'json'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                service = json.loads(result.stdout)
                # Get the internal URL (without external domain)
                url = service['status'].get('url', '')
                
                # Convert external URL to internal ClusterIP format
         
                internal_url = f"http://{self.service_name}.default.svc.cluster.local"
                
                logger.info(f"Service URL: {internal_url}")
                return internal_url
        except Exception as e:
            logger.error(f"Error getting service URL: {e}")
        
        return f"http://{self.service_name}.default.svc.cluster.local"
    
    def get_current_metrics(self):
        """Get current pod count and simulated request rate"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-l', 
                 f'serving.knative.dev/service={self.service_name}',
                 '-o', 'json'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                pods = json.loads(result.stdout)
                current_pods = sum(1 for p in pods['items'] 
                                 if p['status']['phase'] == 'Running')
                
                # Simulate varying request rate
                base_rate = 40
                variation = np.random.normal(0, 15)
                request_rate = max(5, base_rate + variation)
                
                return {
                    'pod_count': current_pods,
                    'request_rate': request_rate
                }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
        
        return {'pod_count': 0, 'request_rate': 10}
    
    def predict_future_load(self):
        """Call forecasting API using kubectl exec"""
        try:
            # Get a running pod
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-l', 
                 f'serving.knative.dev/service={self.service_name}',
                 '-o', 'jsonpath={.items[0].metadata.name}'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                logger.warning("No running pods found")
                return None
            
            pod_name = result.stdout.strip()
            
            # Prepare payload
            recent_data = list(self.history)[-self.lookback:]
            payload = json.dumps({
                'periods': self.prediction_horizon,
                'recent_data': recent_data
            })
            
            # Call API via kubectl exec
            cmd = [
                'kubectl', 'exec', pod_name, '-c', 'forecasting', '--',
                'curl', '-s', '-X', 'POST',
                '-H', 'Content-Type: application/json',
                '-d', payload,
                f'http://localhost:8080/predict/{self.model_type}/{self.function_id}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and result.stdout.strip():
                response = json.loads(result.stdout)
                predictions = response['predictions']
                inference_time = response.get('inference_time_ms', 0)
                
                logger.info(f"{self.model_type.upper()} predictions: "
                          f"{[f'{p:.1f}' for p in predictions[:3]]}... "
                          f"(inference: {inference_time:.1f}ms)")
                
                return predictions
            else:
                logger.error(f"API call failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def calculate_desired_pods(self, predictions):
        """Calculate required pods"""
        if not predictions or len(predictions) == 0:
            return 1
        
        max_predicted_load = max(predictions)
        required_pods = int(np.ceil(max_predicted_load / self.requests_per_pod))
        desired_pods = max(self.min_pods, min(required_pods, self.max_pods))
        
        logger.info(f"Max predicted: {max_predicted_load:.1f} req/min → "
                   f"Desired: {desired_pods} pods")
        
        return desired_pods
    
    def scale_service(self, desired_pods):
        """Scale Knative service"""
        try:
            # Scale by updating minScale annotation
            cmd = [
                'kubectl', 'patch', 'ksvc', self.service_name,
                '--type', 'json',
                '-p', f'[{{"op":"replace","path":"/spec/template/metadata/annotations/autoscaling.knative.dev~1minScale","value":"{max(1, desired_pods)}"}}]'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"Scaled {self.service_name} to min={desired_pods}")
                return True
            else:
                logger.warning(f"Scale failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Scale error: {e}")
            return False
    
    def run(self, interval_seconds=30):
        """Main loop"""

        logger.info(f"PREDICTIVE AUTOSCALER: {self.service_name}")
        logger.info(f"Model: {self.model_type.upper()}")
   
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                logger.info(f"\n[{current_time}] Iteration {iteration}")
   
                
                # Get current state
                metrics = self.get_current_metrics()
                current_pods = metrics['pod_count']
                current_rate = metrics['request_rate']
                
                self.history.append(current_rate)
                
                logger.info(f"Current: {current_pods} pods, {current_rate:.1f} req/min")
                
                # Get predictions
                predictions = self.predict_future_load()
                
                if predictions:
                    desired_pods = self.calculate_desired_pods(predictions)
                    
                    if desired_pods != current_pods:
                        logger.info(f"Scaling: {current_pods} → {desired_pods} pods")
                        self.scale_service(desired_pods)
                    else:
                        logger.info(f"No scaling needed ({desired_pods} pods)")
                else:
                    logger.warning("No predictions, maintaining scale")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("\nStopped")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(interval_seconds)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--function-id', required=True)
    parser.add_argument('--model-type', required=True, 
                       choices=['prophet', 'lstm', 'hybrid'])
    parser.add_argument('--interval', type=int, default=30)
    
    args = parser.parse_args()
    
    autoscaler = PredictiveAutoscaler(
        function_id=args.function_id,
        model_type=args.model_type
    )
    
    autoscaler.run(interval_seconds=args.interval)
