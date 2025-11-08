#!/usr/bin/env python3
"""
Traffic replay using port-forward
"""
import requests
import pandas as pd
import time
import sys

def replay_traffic(function_id, model_type, data_file, duration_minutes=5, speedup=60):
    """Replay traffic using localhost:8080 with Host header"""
    
    service_name = f"{function_id.replace('_', '-')}-{model_type}"
    base_url = "http://localhost:8080"
    host_header = f"{service_name}.default.example.com"
    
    print(f"{'='*60}")
    print(f"TRAFFIC REPLAY: {service_name}")
    print(f"{'='*60}")
    print(f"Base URL: {base_url}")
    print(f"Host header: {host_header}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Speedup: {speedup}x")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(data_file)
    traffic_data = df['y'].values[:duration_minutes]
    
    print(f"Loaded {len(traffic_data)} minutes of traffic")
    print(f"Total requests: {int(traffic_data.sum())}\n")
    
    total_requests = 0
    successful = 0
    failed = 0
    
    for minute, requests_per_min in enumerate(traffic_data):
        requests_per_min = int(requests_per_min)
        
        print(f"Minute {minute+1}/{len(traffic_data)}: {requests_per_min} req/min ", end="", flush=True)
        
        if requests_per_min == 0:
            print("(idle)")
            time.sleep(60 / speedup)
            continue
        
        # Cap requests for testing
        actual_requests = min(requests_per_min, 50)
        delay = (60 / actual_requests) / speedup if actual_requests > 0 else 1
        
        minute_success = 0
        minute_failed = 0
        
        for _ in range(actual_requests):
            try:
                response = requests.get(
                    f"{base_url}/health",
                    headers={'Host': host_header},
                    timeout=5
                )
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
    
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"Total: {total_requests}, Success: {successful}, Failed: {failed}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--function-id', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--speedup', type=int, default=60)
    
    args = parser.parse_args()
    
    replay_traffic(
        args.function_id,
        args.model_type,
        args.data_file,
        args.duration,
        args.speedup
    )
