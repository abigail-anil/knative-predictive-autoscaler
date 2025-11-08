#!/usr/bin/env python3
"""
Replay actual traffic patterns from CSV data
"""
import requests
import pandas as pd
import time
import subprocess
import json
import sys
from datetime import datetime

def get_service_url(service_name):
    """Get Knative service URL"""
    result = subprocess.run(
        ['kubectl', 'get', 'ksvc', service_name, '-o', 'json'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        service = json.loads(result.stdout)
        return service['status']['url']
    return None

def replay_traffic(function_id, model_type, data_file, duration_minutes=5, speedup=60):
    """
    Replay traffic pattern
    
    Args:
        function_id: e.g., 'func_235'
        model_type: e.g., 'prophet'
        data_file: Path to CSV
        duration_minutes: How many minutes of data to replay
        speedup: Speed multiplier (60 = 1 hour becomes 1 minute)
    """
    
    service_name = f"{function_id.replace('_', '-')}-{model_type}"
    service_url = get_service_url(service_name)
    
    if not service_url:
        print(f"Could not get URL for {service_name}")
        return
    
 
    print(f"TRAFFIC REPLAY: {service_name}")

    print(f"Service URL: {service_url}")
    print(f"Data file: {data_file}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Speedup: {speedup}x")
    print(f"\n")
    
    # Load data
    df = pd.read_csv(data_file)
    traffic_data = df['y'].values[:duration_minutes]
    
    print(f"Loaded {len(traffic_data)} minutes of traffic data")
    print(f"Total requests to send: {int(traffic_data.sum())}\n")
    
    total_requests = 0
    successful = 0
    failed = 0
    
    for minute, requests_per_min in enumerate(traffic_data):
        requests_per_min = int(requests_per_min)
        
        print(f"Minute {minute+1}/{len(traffic_data)}: {requests_per_min} req/min ", end="")
        
        if requests_per_min == 0:
            print("(idle)")
            time.sleep(60 / speedup)
            continue
        
        # Send requests for this minute
        minute_success = 0
        minute_failed = 0
        
        # Cap at 50 requests per minute for testing
        actual_requests = min(requests_per_min, 50)
        delay = (60 / actual_requests) / speedup
        
        for _ in range(actual_requests):
            try:
                response = requests.get(f"{service_url}/health", timeout=5)
                if response.status_code == 200:
                    minute_success += 1
                else:
                    minute_failed += 1
            except:
                minute_failed += 1
            
            time.sleep(delay)
        
        total_requests += actual_requests
        successful += minute_success
        failed += minute_failed
        
        print(f"→ {minute_success}/{actual_requests} success")
    
    print(f"\n")
    print(f"TRAFFIC REPLAY COMPLETED")
    
    print(f"Total requests: {total_requests}")
    print(f"Successful: {successful} ({successful/total_requests*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_requests*100:.1f}%)")
    print(f"\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--function-id', required=True, help='e.g., func_235')
    parser.add_argument('--model-type', required=True, help='e.g., prophet, lstm, hybrid')
    parser.add_argument('--data-file', required=True, help='Path to CSV')
    parser.add_argument('--duration', type=int, default=5, help='Minutes to replay')
    parser.add_argument('--speedup', type=int, default=60, help='Speed multiplier')
    
    args = parser.parse_args()
    
    replay_traffic(
        args.function_id,
        args.model_type,
        args.data_file,
        args.duration,
        args.speedup
    )
