#!/bin/bash

ACR="${ACR_LOGIN_SERVER:-knativeacr25446.azurecr.io}"

FUNCTIONS=("func_126")
MODELS=("prophet" "lstm" "hybrid")

mkdir -p knative

for func in "${FUNCTIONS[@]}"; do
  for model in "${MODELS[@]}"; do

    SERVICE_NAME="${func//_/-}-${model}"

    MIN_SCALE="1"
    MAX_SCALE="3"
    TARGET="5"

    if [[ "$model" == "prophet" ]]; then
        REQ_MEM="1Gi"
        LIM_MEM="2Gi"
        TIMEOUT="300"
        
    elif [[ "$model" == "lstm" ]]; then
        REQ_MEM="1.5Gi" 
        LIM_MEM="3Gi"
        TIMEOUT="60"

    elif [[ "$model" == "hybrid" ]]; then
        REQ_MEM="1.1Gi"
        LIM_MEM="1.8Gi"
        TIMEOUT="300"
    fi   

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
        autoscaling.knative.dev/min-scale: "${MIN_SCALE}"
        autoscaling.knative.dev/max-scale: "${MAX_SCALE}"
        autoscaling.knative.dev/target: "${TARGET}"
        autoscaling.knative.dev/metric: "concurrency"
        autoscaling.knative.dev/window: "10s"
        autoscaling.knative.dev/scale-down-delay: "30s"
        autoscaling.knative.dev/stable-window: "30s"
        serving.knative.dev/response-start-timeout: "300"

    spec:
      containerConcurrency: 1
      timeoutSeconds: 60

      containers:
      - image: ${ACR}/forecasting-api:v44
        imagePullPolicy: Always
        name: forecasting
        env:
        - name: MODEL_TYPE
          value: "${model}"
        - name: FUNCTION_ID
          value: "${func}"
        ${PROPHET_ENV}
        ports:
        - containerPort: 8080

        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 3
          periodSeconds: 5

        resources:
          requests:
            cpu: "500m"
            memory: "${REQ_MEM}"
          limits:
            cpu: "1500m"
            memory: "${LIM_MEM}"
YAML

    echo "Generated knative/${SERVICE_NAME}.yaml with memory ${REQ_MEM} -> ${LIM_MEM}"

  done
done
