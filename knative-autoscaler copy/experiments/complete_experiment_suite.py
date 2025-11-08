#!/usr/bin/env python3
"""
Complete Experiment Suite
Tests all functions across all models (Prophet, LSTM, Hybrid, Reactive)
"""
import subprocess
import json
import sys
import time
from datetime import datetime

# Configuration
EXPERIMENTS = [
    # Function, Model, Duration (minutes)
    ('func_235', 'prophet', 10),
    ('func_235', 'lstm', 10),
    ('func_235', 'hybrid', 10),
    ('func_235', 'reactive', 10),
    
    ('func_126', 'prophet', 10),
    ('func_126', 'lstm', 10),
    ('func_126', 'hybrid', 10),
    ('func_126', 'reactive', 10),
    
    ('func_110', 'prophet', 10),
    ('func_110', 'lstm', 10),
    ('func_110', 'hybrid', 10),
    ('func_110', 'reactive', 10),
]

COOLDOWN_MINUTES = 2

def run_experiment(function_id, model_type, duration_minutes):
    """Run a single experiment"""
    
    service_name = f"{function_id.replace('_', '-')}-{model_type}"
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {service_name}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"{'='*60}")
    
    # Check if service exists and is ready
    result = subprocess.run(
        ['kubectl', 'get', 'ksvc', service_name, '-o', 'json'],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Service {service_name} not found, skipping")
        return False
    
    service = json.loads(result.stdout)
    ready = any(
        cond.get('status') == 'True' 
        for cond in service.get('status', {}).get('conditions', []) 
        if cond.get('type') == 'Ready'
    )
    
    if not ready:
        print(f"Service {service_name} not ready, skipping")
        return False
    
    # Start metrics collector
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"../results/logs/{service_name}_{timestamp}.log"
    
    metrics_cmd = [
        'python3', 'metrics_collector.py',
        '--function-id', function_id,
        '--model-type', model_type,
        '--duration', str(duration_minutes * 60),
        '--interval', '5'
    ]
    
    print(f"Starting metrics collector...")
    with open(log_file, 'w') as f:
        metrics_process = subprocess.Popen(
            metrics_cmd,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    
    time.sleep(5)
    
    # Run traffic replay
    print(f"Starting traffic replay...")
    traffic_cmd = [
        'python3', 'traffic_generator.py',
        '--function-id', function_id,
        '--model-type', model_type,
        '--data-file', f'../data/timeseries_{function_id}.csv',
        '--duration', str(duration_minutes),
        '--speedup', '60'
    ]
    
    traffic_result = subprocess.run(traffic_cmd)
    
    # Wait for metrics to finish
    print(f"Waiting for metrics collector...")
    metrics_process.wait(timeout=duration_minutes * 60 + 30)
    
    print(f"Experiment completed: {service_name}")
    print(f"  Logs: {log_file}")
    
    return traffic_result.returncode == 0

def main():
    print("COMPLETE EXPERIMENT SUITE")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Estimated time: {sum(e[2] for e in EXPERIMENTS) + (len(EXPERIMENTS)-1) * COOLDOWN_MINUTES} minutes")
    
    completed = 0
    skipped = 0
    failed = 0
    
    for i, (function_id, model_type, duration) in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}]")
        
        success = run_experiment(function_id, model_type, duration)
        
        if success:
            completed += 1
        elif success is False:
            skipped += 1
        else:
            failed += 1
        
        # Cooldown between experiments
        if i < len(EXPERIMENTS) - 1:
            print(f"\nCooldown: {COOLDOWN_MINUTES} minutes...")
            time.sleep(COOLDOWN_MINUTES * 60)
    
    print("\n")
    print("EXPERIMENT SUITE COMPLETED")

    print(f"Completed: {completed}/{len(EXPERIMENTS)}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"\nResults saved in: ../results/")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
