"""
Traffic Generator - Replay actual Azure Functions invocation patterns
"""
import pandas as pd
import numpy as np
import requests
import time
import threading
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficGenerator:
    def __init__(self, service_url, function_id, data_file):
        """
        Initialize traffic generator
        
        Args:
            service_url: Knative service URL (e.g., http://func-235.default.example.com)
            function_id: Function ID (e.g., 'func_235')
            data_file: Path to timeseries CSV file
        """
        self.service_url = service_url
        self.function_id = function_id
        self.data_file = data_file
        self.running = False
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
        
    def load_traffic_pattern(self):
        """Load actual traffic pattern from CSV"""
        logger.info(f"Loading traffic pattern from {self.data_file}")
        
        df = pd.read_csv(self.data_file)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Get the 'y' column (requests per minute)
        self.traffic_pattern = df['y'].values
        
        logger.info(f"Loaded {len(self.traffic_pattern)} time periods")
        logger.info(f"  Total requests: {self.traffic_pattern.sum():,.0f}")
        logger.info(f"  Max req/min: {self.traffic_pattern.max():.0f}")
        logger.info(f"  Avg req/min: {self.traffic_pattern.mean():.2f}")
        
        return self.traffic_pattern
    
    def send_request(self):
        """Send a single HTTP request to the service"""
        try:
            response = requests.get(
                self.service_url,
                timeout=30
            )
            
            if response.status_code == 200:
                self.stats['successful'] += 1
                return True
            else:
                self.stats['failed'] += 1
                return False
                
        except Exception as e:
            self.stats['failed'] += 1
            logger.debug(f"Request failed: {str(e)}")
            return False
    
    def replay_pattern(self, speedup_factor=1.0, duration_minutes=None):
        """
        Replay actual traffic pattern
        
        Args:
            speedup_factor: Speed multiplier (1.0 = real-time, 2.0 = 2x faster)
            duration_minutes: How long to replay (None = entire pattern)
        """
        self.running = True
        self.stats['start_time'] = time.time()
        
        pattern = self.load_traffic_pattern()
        
        # Determine how many minutes to replay
        if duration_minutes:
            pattern = pattern[:duration_minutes]
        
        logger.info(f"\nStarting traffic replay:")
        logger.info(f"  Pattern length: {len(pattern)} minutes")
        logger.info(f"  Speedup factor: {speedup_factor}x")
        logger.info(f"  Target: {self.service_url}")
        logger.info(f"="*60)
        
        for minute_idx, requests_per_minute in enumerate(pattern):
            if not self.running:
                break
            
            # How many requests to send this minute
            num_requests = int(requests_per_minute)
            
            if num_requests == 0:
                logger.info(f"Minute {minute_idx+1}/{len(pattern)}: 0 requests (idle)")
                time.sleep(60 / speedup_factor)
                continue
            
            logger.info(f"Minute {minute_idx+1}/{len(pattern)}: Sending {num_requests} requests...")
            
            # Calculate inter-request delay (spread requests across the minute)
            delay_between_requests = (60 / num_requests) / speedup_factor
            
            # Send requests for this minute
            threads = []
            for _ in range(num_requests):
                if not self.running:
                    break
                
                # Send request in background thread
                t = threading.Thread(target=self.send_request)
                t.start()
                threads.append(t)
                
                self.stats['total_requests'] += 1
                
                # Wait before next request
                time.sleep(delay_between_requests)
            
            # Wait for all threads to complete
            for t in threads:
                t.join(timeout=5)
            
            # Log progress
            if (minute_idx + 1) % 10 == 0:
                success_rate = (self.stats['successful'] / self.stats['total_requests'] * 100
                               if self.stats['total_requests'] > 0 else 0)
                logger.info(f"  Progress: {minute_idx+1}/{len(pattern)} minutes, "
                           f"Success rate: {success_rate:.1f}%")
        
        self.stats['end_time'] = time.time()
        self.print_summary()
    
    def generate_synthetic_burst(self, base_load=10, burst_load=100, 
                                 burst_duration_min=5, total_duration_min=30):
        """
        Generate synthetic traffic with bursts
        
        Args:
            base_load: Requests per minute during normal operation
            burst_load: Requests per minute during burst
            burst_duration_min: Duration of each burst
            total_duration_min: Total test duration
        """
        self.running = True
        self.stats['start_time'] = time.time()
        
        logger.info(f"\nStarting synthetic burst traffic:")
        logger.info(f"  Base load: {base_load} req/min")
        logger.info(f"  Burst load: {burst_load} req/min")
        logger.info(f"  Burst duration: {burst_duration_min} min")
        logger.info(f"  Total duration: {total_duration_min} min")
        logger.info(f"="*60)
        
        # Create pattern: base -> burst -> base -> burst
        pattern = []
        for minute in range(total_duration_min):
            # Burst every 10 minutes
            if (minute // 10) % 2 == 1 and (minute % 10) < burst_duration_min:
                pattern.append(burst_load)
            else:
                pattern.append(base_load)
        
        # Replay this synthetic pattern
        for minute_idx, requests_per_minute in enumerate(pattern):
            if not self.running:
                break
            
            num_requests = requests_per_minute
            delay_between_requests = 60 / num_requests if num_requests > 0 else 60
            
            logger.info(f"Minute {minute_idx+1}/{total_duration_min}: "
                       f"{'BURST' if requests_per_minute == burst_load else 'BASE'} "
                       f"({num_requests} req/min)")
            
            for _ in range(num_requests):
                if not self.running:
                    break
                threading.Thread(target=self.send_request).start()
                self.stats['total_requests'] += 1
                time.sleep(delay_between_requests)
        
        self.stats['end_time'] = time.time()
        self.print_summary()
    
    def stop(self):
        """Stop traffic generation"""
        logger.info("\nStopping traffic generator...")
        self.running = False
    
    def print_summary(self):
        """Print traffic generation summary"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        logger.info(f"\n{'='*60}")
        logger.info("TRAFFIC GENERATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Function: {self.function_id}")
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Total requests: {self.stats['total_requests']:,}")
        logger.info(f"Successful: {self.stats['successful']:,}")
        logger.info(f"Failed: {self.stats['failed']:,}")
        
        if self.stats['total_requests'] > 0:
            success_rate = self.stats['successful'] / self.stats['total_requests'] * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
            
            avg_rps = self.stats['total_requests'] / duration
            logger.info(f"Average RPS: {avg_rps:.2f}")
        
        logger.info(f"{'='*60}\n")

# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate traffic for Knative service')
    parser.add_argument('--service-url', required=True, help='Knative service URL')
    parser.add_argument('--function-id', required=True, help='Function ID (e.g., func_235)')
    parser.add_argument('--data-file', required=True, help='Path to timeseries CSV')
    parser.add_argument('--mode', choices=['replay', 'synthetic'], default='replay',
                       help='Traffic generation mode')
    parser.add_argument('--speedup', type=float, default=10.0,
                       help='Speedup factor for replay (default: 10x)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Duration in minutes (None = full replay)')
    
    args = parser.parse_args()
    
    generator = TrafficGenerator(
        service_url=args.service_url,
        function_id=args.function_id,
        data_file=args.data_file
    )
    
    try:
        if args.mode == 'replay':
            generator.replay_pattern(
                speedup_factor=args.speedup,
                duration_minutes=args.duration
            )
        else:
            generator.generate_synthetic_burst(
                base_load=10,
                burst_load=100,
                burst_duration_min=5,
                total_duration_min=30
            )
    except KeyboardInterrupt:
        generator.stop()
        logger.info("\nTraffic generation interrupted by user")