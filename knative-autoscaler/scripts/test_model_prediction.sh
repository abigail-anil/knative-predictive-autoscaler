#!/bin/bash

FUNCTION_ID=$1
MODEL_TYPE=$2

if [ -z "$FUNCTION_ID" ] || [ -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 <function_id> <model_type>"
    echo "Example: $0 func_235 prophet"
    exit 1
fi

SERVICE_NAME="${FUNCTION_ID//_/-}-${MODEL_TYPE}"

echo "Testing model prediction for: $SERVICE_NAME"
echo "==========================================="

# Get pod name
POD_NAME=$(kubectl get pods -l serving.knative.dev/service=$SERVICE_NAME -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "No running pods found for $SERVICE_NAME"
    echo "Creating a pod by accessing the service..."
    
    # Trigger pod creation
    kubectl run test-client --rm -i --restart=Never --image=curlimages/curl -- \
      curl -s http://$SERVICE_NAME.default.svc.cluster.local/health
    
    sleep 10
    
    POD_NAME=$(kubectl get pods -l serving.knative.dev/service=$SERVICE_NAME -o jsonpath='{.items[0].metadata.name}')
fi

echo "Pod: $POD_NAME"
echo ""

# Test health
echo "1. Testing health endpoint..."
kubectl exec $POD_NAME -c forecasting -- curl -s http://localhost:8080/health | jq .

echo ""
echo "2. Testing prediction endpoint..."

# Prepare test data
TEST_DATA='{"periods":5,"recent_data":[100,105,110,95,120,115,108,130,125,140,135,145,150,148,155,160,158,165,170,168,175,180,178,185,190,188,195,200,198,205]}'

kubectl exec $POD_NAME -c forecasting -- \
  curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "$TEST_DATA" \
  http://localhost:8080/predict/$MODEL_TYPE/$FUNCTION_ID | jq .

echo ""
echo "==========================================="
echo "Test completed"