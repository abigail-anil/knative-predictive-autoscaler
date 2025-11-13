#!/bin/bash

FUNCTIONS=("func_235" "func_126" "func_110")
MODELS=("prophet" "lstm" "hybrid")

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
        autoscaling.knative.dev/min-scale: "1"
        autoscaling.knative.dev/max-scale: "10"
        autoscaling.knative.dev/target: "50"
        autoscaling.knative.dev/metric: "concurrency"
        autoscaling.knative.dev/queue-sidecar-cpu-request: "50m"
        autoscaling.knative.dev/queue-sidecar-memory-request: "64Mi"
        autoscaling.knative.dev/queue-sidecar-cpu-limit: "100m"
        autoscaling.knative.dev/queue-sidecar-memory-limit: "128Mi"
    spec:
      containers:
      - image: ${ACR_LOGIN_SERVER}/forecasting-api:v3
        imagePullPolicy: Always
        name: forecasting
        env:
        - name: MODEL_TYPE
          value: "${model}"
        - name: FUNCTION_ID
          value: "${func}"
        ports:
        - containerPort: 8080
          protocol: TCP
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1
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
done

echo "All predictive YAML files generated with readiness probe and queue-proxy resource limits"
