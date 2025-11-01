#!/bin/bash

# Configuration
HOST="hello-world.default.127.0.0.1.sslip.io"
KOURIER_PORT=8080
DURATION=60  # seconds
CONCURRENT=10

echo "Starting load test..."
echo "Duration: ${DURATION}s"
echo "Concurrent requests: ${CONCURRENT}"
echo "Target: ${HOST}"
echo ""

# Function to send requests
send_requests() {
    while true; do
        curl -s -H "Host: ${HOST}" http://localhost:${KOURIER_PORT} > /dev/null
        sleep 0.1  # 10 requests per second per worker
    done
}

# Start concurrent workers
for i in $(seq 1 $CONCURRENT); do
    send_requests &
done

# Run for specified duration
sleep $DURATION

# Stop all background jobs
pkill -P $$

echo "Load test completed!"