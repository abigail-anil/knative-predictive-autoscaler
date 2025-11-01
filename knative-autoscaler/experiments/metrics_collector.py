"""
Comprehensive Metrics Collector for Knative Autoscaling Experiments
Collects: Pod metrics, Prediction accuracy, Scaling decisions, Resource utilization
"""
import subprocess
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, function_id, model_type, output_dir='results/metrics'):
        """
        Initialize metrics collector
        
        Args:
            function_id: Function being tested (e.g., 'func_235')
            model_type: Model type ('prophet', 'lstm', 'hybrid', 'reactive')
            output_dir: Directory to save metrics
        """
        self.function_id = function_id
        self.model_type = model_type
        self.output_dir = output_dir
        self.service_name = f"{function_id}-{model_type}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = []
        self.running = False
        
    def get_pod_metrics(self):
        """Get current pod count and status"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-l', f'serving.knative.dev/service={self.service_name}',
                 '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pods = json.loads(result.stdout)
                
                running_pods = sum(1 for pod in pods['items'] 
                                 if pod['status']['phase'] == 'Running')
                pending_pods = sum(1 for pod in pods['items'] 
                                 if pod['status']['phase'] == 'Pending')
                
                # Get pod start times for cold start detection
                pod_ages = []
                for pod in pods['items']:
                    if pod['status']['phase'] == 'Running':
                        start_time = pod['status'].get('startTime')
                        if start_time:
                            pod_ages.append((datetime.utcnow() - 
                                           datetime.fromisoformat(start_time.replace('Z', '+00:00')))
                                          .total_seconds())
                
                return {
                    'pod_count': running_pods,
                    'pending_pods': pending_pods,
                    'avg_pod_age': np.mean(pod_ages) if pod_ages else 0,
                    'min_pod_age': min(pod_ages) if pod_ages else 0
                }
            
            return {'pod_count': 0, 'pending_pods': 0, 'avg_pod_age': 0, 'min_pod_age': 0}
            
        except Exception as e:
            logger.error(f"Error getting pod metrics: {str(e)}")
            return {'pod_count': 0, 'pending_pods': 0, 'avg_pod_age': 0, 'min_pod_age': 0}
    
    def get_resource_usage(self):
        """Get CPU and memory usage"""
        try:
            result = subprocess.run(
                ['kubectl', 'top', 'pods', '-l', f'serving.knative.dev/service={self.service_name}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                cpu_usage = []
                memory_usage = []
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Parse CPU (e.g., "100m" -> 100)
                        cpu = parts[1].replace('m', '')
                        try:
                            cpu_usage.append(float(cpu))
                        except:
                            pass
                        
                        # Parse memory (e.g., "256Mi" -> 256)
                        memory = parts[2].replace('Mi', '').replace('Gi', '000')
                        try:
                            memory_usage.append(float(memory))
                        except:
                            pass
                
                return {
                    'cpu_millicores': np.mean(cpu_usage) if cpu_usage else 0,
                    'memory_mb': np.mean(memory_usage) if memory_usage else 0,
                    'total_cpu': sum(cpu_usage),
                    'total_memory': sum(memory_usage)
                }
            
            return {'cpu_millicores': 0, 'memory_mb': 0, 'total_cpu': 0, 'total_memory': 0}
            
        except Exception as e:
            logger.debug(f"Error getting resource usage: {str(e)}")
            return {'cpu_millicores': 0, 'memory_mb': 0, 'total_cpu': 0, 'total_memory': 0}
    
    def get_knative_metrics(self):
        """Get Knative autoscaler metrics"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'podautoscaler', self.service_name, '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pa = json.loads(result.stdout)
                status = pa.get('status', {})
                
                return {
                    'desired_scale': status.get('desiredScale', 0),
                    'actual_scale': status.get('actualScale', 0),
                    'requested_scale': status.get('requestedScale', 0)
                }
            
            return {'desired_scale': 0, 'actual_scale': 0, 'requested_scale': 0}
            
        except Exception as e:
            logger.debug(f"Error getting Knative metrics: {str(e)}")
            return {'desired_scale': 0, 'actual_scale': 0, 'requested_scale': 0}
    
    def detect_cold_start(self, prev_pod_count, current_pod_count, min_pod_age):
        """
        Detect cold start events
        Returns: (is_cold_start, cold_start_time)
        """
        # Cold start if:
        # 1. New pod appeared (count increased)
        # 2. Pod is very young (< 30 seconds)
        
        if current_pod_count > prev_pod_count and min_pod_age < 30:
            return True, min_pod_age
        
        return False, 0
    
    def collect_metrics(self, duration_seconds=300, interval_seconds=2):
        """
        Collect metrics for specified duration
        
        Args:
            duration_seconds: How long to collect
            interval_seconds: Collection interval
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting metrics collection for {self.function_id} ({self.model_type})")
        logger.info(f"Duration: {duration_seconds}s, Interval: {interval_seconds}s")
        logger.info(f"{'='*60}\n")
        
        self.running = True
        start_time = time.time()
        prev_pod_count = 0
        
        # Metrics header
        logger.info(f"{'Time':<10} {'Pods':<6} {'Pending':<8} {'Desired':<8} "
                   f"{'CPU(m)':<8} {'Mem(MB)':<8} {'Cold Start':<12}")
        logger.info("-" * 70)
        
        while self.running and (time.time() - start_time) < duration_seconds:
            timestamp = time.time()
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Collect all metrics
            pod_metrics = self.get_pod_metrics()
            resource_metrics = self.get_resource_usage()
            knative_metrics = self.get_knative_metrics()
            
            # Detect cold starts
            is_cold_start, cold_start_time = self.detect_cold_start(
                prev_pod_count,
                pod_metrics['pod_count'],
                pod_metrics['min_pod_age']
            )
            
            # Compile metrics
            metric_entry = {
                'timestamp': timestamp,
                'time': current_time,
                'pod_count': pod_metrics['pod_count'],
                'pending_pods': pod_metrics['pending_pods'],
                'desired_scale': knative_metrics['desired_scale'],
                'actual_scale': knative_metrics['actual_scale'],
                'requested_scale': knative_metrics['requested_scale'],
                'cpu_millicores': resource_metrics['cpu_millicores'],
                'memory_mb': resource_metrics['memory_mb'],
                'total_cpu': resource_metrics['total_cpu'],
                'total_memory': resource_metrics['total_memory'],
                'avg_pod_age': pod_metrics['avg_pod_age'],
                'is_cold_start': is_cold_start,
                'cold_start_time': cold_start_time if is_cold_start else 0
            }
            
            self.metrics.append(metric_entry)
            
            # Log current state
            cold_start_indicator = f"YES ({cold_start_time:.1f}s)" if is_cold_start else ""
            logger.info(f"{current_time:<10} {pod_metrics['pod_count']:<6} "
                       f"{pod_metrics['pending_pods']:<8} "
                       f"{knative_metrics['desired_scale']:<8} "
                       f"{resource_metrics['cpu_millicores']:<8.0f} "
                       f"{resource_metrics['memory_mb']:<8.0f} "
                       f"{cold_start_indicator:<12}")
            
            prev_pod_count = pod_metrics['pod_count']
            
            # Wait for next collection
            time.sleep(interval_seconds)
        
        self.save_metrics()
        self.print_summary()
    
    def save_metrics(self):
        """Save collected metrics to CSV"""
        if not self.metrics:
            logger.warning("No metrics to save")
            return
        
        df = pd.DataFrame(self.metrics)
        
        filename = f"{self.output_dir}/{self.function_id}_{self.model_type}_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        
        logger.info(f"\n✓ Metrics saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print metrics summary"""
        if not self.metrics:
            return
        
        df = pd.DataFrame(self.metrics)
        
        logger.info(f"\n{'='*60}")
        logger.info("METRICS SUMMARY")
        logger.info(f"{'='*60}")
        
        logger.info(f"\nPod Scaling:")
        logger.info(f"  Max pods: {df['pod_count'].max()}")
        logger.info(f"  Avg pods: {df['pod_count'].mean():.2f}")
        logger.info(f"  Min pods: {df['pod_count'].min()}")
        logger.info(f"  Scale-to-zero events: {(df['pod_count'] == 0).sum()}")
        
        logger.info(f"\nCold Starts:")
        cold_starts = df[df['is_cold_start'] == True]
        logger.info(f"  Total cold starts: {len(cold_starts)}")
        if len(cold_starts) > 0:
            logger.info(f"  Avg cold start time: {cold_starts['cold_start_time'].mean():.2f}s")
            logger.info(f"  Max cold start time: {cold_starts['cold_start_time'].max():.2f}s")
        
        logger.info(f"\nResource Usage:")
        logger.info(f"  Avg CPU: {df['cpu_millicores'].mean():.0f} millicores")
        logger.info(f"  Peak CPU: {df['cpu_millicores'].max():.0f} millicores")
        logger.info(f"  Avg Memory: {df['memory_mb'].mean():.0f} MB")
        logger.info(f"  Peak Memory: {df['memory_mb'].max():.0f} MB")
        
        logger.info(f"\nScaling Accuracy:")
        df['scale_diff'] = abs(df['desired_scale'] - df['actual_scale'])
        logger.info(f"  Avg scale difference: {df['scale_diff'].mean():.2f}")
        logger.info(f"  Perfect scaling: {(df['scale_diff'] == 0).sum() / len(df) * 100:.1f}%")
        
        logger.info(f"{'='*60}\n")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False

# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect Knative metrics')
    parser.add_argument('--function-id', required=True, help='Function ID')
    parser.add_argument('--model-type', required=True, 
                       choices=['prophet', 'lstm', 'hybrid', 'reactive'],
                       help='Model type')
    parser.add_argument('--duration', type=int, default=300,
                       help='Collection duration in seconds')
    parser.add_argument('--interval', type=int, default=2,
                       help='Collection interval in seconds')
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.function_id, args.model_type)
    
    try:
        collector.collect_metrics(
            duration_seconds=args.duration,
            interval_seconds=args.interval
        )
    except KeyboardInterrupt:
        collector.stop()
        logger.info("\nMetrics collection stopped by user")