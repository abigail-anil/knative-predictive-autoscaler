#!/bin/bash

FUNCTIONS=("func_235" "func_126" "func_110")
IMAGE_SHA=$(docker inspect forecasting-api:latest --format='{{.Id}}' | cut -d':' -f2)

mkdir -p knative

for func in "${FUNCTIONS[@]}"; do
    SERVICE_NAME="${func//_/-}-reactive"
    
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
        # NO prediction - pure reactive scaling
    spec:
      containers:
      - image: forecasting-api@sha256:${IMAGE_SHA}
        imagePullPolicy: Never
        name: forecasting
        env:
        - name: MODEL_TYPE
          value: "reactive"
        - name: FUNCTION_ID
          value: "${func}"
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
YAML
    
    echo "Created knative/${SERVICE_NAME}.yaml"
done

echo "Reactive services created"
