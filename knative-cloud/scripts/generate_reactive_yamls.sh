#!/bin/bash
# Reactive autoscaling YAML generator for AKS + Knative

ACR="${ACR_LOGIN_SERVER:-knativeacr25446.azurecr.io}"


FUNCTIONS=("func_235" "func_126" "func_110")
MODELS=("reactive")

mkdir -p knative/reactive

for func in "${FUNCTIONS[@]}"; do
    for model in "${MODELS[@]}"; do
        SERVICE_NAME="${func//_/-}-${model}"

        cat > knative/reactive/${SERVICE_NAME}.yaml << YAML
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  namespace: default
spec:
  template:
    metadata:
      annotations:
        # --- Reactive autoscaling configuration ---
        autoscaling.knative.dev/class: "kpa.autoscaling.knative.dev"
        autoscaling.knative.dev/metric: "concurrency"
        autoscaling.knative.dev/target: "1"                      # scale aggressively
        autoscaling.knative.dev/window: "6s"                     # shorter averaging window
        autoscaling.knative.dev/min-scale: "0"                   # allow scale-to-zero
        autoscaling.knative.dev/max-scale: "3"                   # cap pods to control cost
        autoscaling.knative.dev/scale-to-zero-grace-period: "10s"
        autoscaling.knative.dev/scale-to-zero-pod-retention-period: "10s"

        # --- Queue proxy limits for AKS quota compliance ---
        autoscaling.knative.dev/queue-sidecar-cpu-request: "50m"
        autoscaling.knative.dev/queue-sidecar-memory-request: "64Mi"
        autoscaling.knative.dev/queue-sidecar-cpu-limit: "100m"
        autoscaling.knative.dev/queue-sidecar-memory-limit: "128Mi"

    spec:
      containers:
      - image: ${ACR}/forecasting-api:v44
        imagePullPolicy: IfNotPresent
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

        echo "Created knative/reactive/${SERVICE_NAME}.yaml with aggressive autoscaling"
    done
done

echo "All reactive YAML files generated under knative/reactive/"
