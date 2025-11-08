#!/bin/bash

echo "1. Docker images:"
docker images | grep forecasting-api
echo ""
echo "2. Knative services:"
kubectl get ksvc
echo ""
echo "3. Pods:"
kubectl get pods
echo ""
echo "4. Revisions:"
kubectl get revision
echo ""
echo "5. Events (last 20):"
kubectl get events --sort-by='.lastTimestamp' | tail -20
echo ""
echo "6. Describe one service:"
kubectl describe ksvc func-235-prophet | tail -50
