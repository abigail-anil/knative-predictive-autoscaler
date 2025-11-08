#!/bin/bash

FUNCTION_ID=$1
MODEL_TYPE=$2
DURATION=${3:-10}  # minutes

if [ -z "$FUNCTION_ID" ] || [ -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 <function_id> <model_type> [duration]"
    echo "Example: $0 func_235 prophet 10"
    exit 1
fi

SERVICE_NAME="${FUNCTION_ID//_/-}-${MODEL_TYPE}"


echo "PREDICTIVE AUTOSCALING EXPERIMENT"

echo "Function: $FUNCTION_ID"
echo "Model: $MODEL_TYPE"
echo "Duration: $DURATION minutes"
echo "==========================================="

# 1. Start predictive autoscaler in background
echo "Starting predictive autoscaler..."
cd ../autoscalers
python3 unified_predictive_autoscaler.py \
  --function-id $FUNCTION_ID \
  --model-type $MODEL_TYPE \
  --interval 30 \
  > ../results/logs/autoscaler_${SERVICE_NAME}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

AUTOSCALER_PID=$!
echo "Autoscaler PID: $AUTOSCALER_PID"

cd ../experiments
sleep 5

# 2. Start metrics collector
echo "Starting metrics collector..."
python3 metrics_collector.py \
  --function-id $FUNCTION_ID \
  --model-type $MODEL_TYPE \
  --duration $((DURATION * 60)) \
  --interval 5 \
  > ../results/logs/metrics_${SERVICE_NAME}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

METRICS_PID=$!
echo "Metrics PID: $METRICS_PID"

sleep 5

# 3. Generate traffic
echo "Starting traffic generation..."
python3 replay_traffic.py \
  --function-id $FUNCTION_ID \
  --model-type $MODEL_TYPE \
  --data-file ../data/timeseries_${FUNCTION_ID}.csv \
  --duration $DURATION \
  --speedup 60

# 4. Wait for metrics to finish
echo "Waiting for metrics collector..."
wait $METRICS_PID

# 5. Stop autoscaler
echo "Stopping autoscaler..."
kill $AUTOSCALER_PID

echo "Experiment completed: $SERVICE_NAME"
