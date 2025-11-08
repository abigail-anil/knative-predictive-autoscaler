#!/bin/bash

FUNCTIONS=("func_235" "func_126" "func_110")
MODELS=("prophet" "lstm" "hybrid" "reactive")

# Get image SHA
IMAGE_SHA=$(docker inspect forecasting-api:latest --format='{{.Id}}' | cut -d':' -f2)
IMAGE_FULL="forecasting-api@sha256:${IMAGE_SHA}"

echo "Using image: $IMAGE_FULL"

mkdir -p knative

for func in "${FUNCTIONS[@]}"; do
    for model in "${MODELS[@]}"; do
        SERVICE_NAME="${func//_/-}-${model}"
        
        cat > knative/${SERVICE_NAME}.yaml << YAML
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  namespace: default
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/min-scale: "0"
        autoscaling.knative.dev/max-scale: "10"
        autoscaling.knative.dev/target: "10"
        autoscaling.knative.dev/metric: "concurrency"
    spec:
      containers:
      - image: ${IMAGE_FULL}
        imagePullPolicy: Never
        name: forecasting
        env:
        - name: MODEL_TYPE
          value: "${model}"
        - name: FUNCTION_ID
          value: "${func}"
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
YAML
        
        echo "Created knative/${SERVICE_NAME}.yaml"
    done
done

echo "All YAML files generated with image SHA"
