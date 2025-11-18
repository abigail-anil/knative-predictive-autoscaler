#!/bin/bash

# Check current Knative service configuration for autoscaling issues

SERVICE_NAME="func-235-prophet"

echo "========================================"
echo "KNATIVE SERVICE DIAGNOSTICS"
echo "Service: $SERVICE_NAME"
echo "========================================"

# Check if service exists
echo ""
echo "1. Checking if service exists..."
kubectl get ksvc $SERVICE_NAME &>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ Service not found!"
    echo "   Deploy with: kubectl apply -f knative/"
    exit 1
fi
echo "   ✓ Service exists"

# Get autoscaling annotations
echo ""
echo "2. Current autoscaling settings:"
echo "   ─────────────────────────────"

MIN_SCALE=$(kubectl get ksvc $SERVICE_NAME -o jsonpath='{.spec.template.metadata.annotations.autoscaling\.knative\.dev/min-scale}')
MAX_SCALE=$(kubectl get ksvc $SERVICE_NAME -o jsonpath='{.spec.template.metadata.annotations.autoscaling\.knative\.dev/max-scale}')
TARGET=$(kubectl get ksvc $SERVICE_NAME -o jsonpath='{.spec.template.metadata.annotations.autoscaling\.knative\.dev/target}')
METRIC=$(kubectl get ksvc $SERVICE_NAME -o jsonpath='{.spec.template.metadata.annotations.autoscaling\.knative\.dev/metric}')

echo "   min-scale: ${MIN_SCALE:-NOT SET}"
echo "   max-scale: ${MAX_SCALE:-NOT SET}"
echo "   target: ${TARGET:-NOT SET}"
echo "   metric: ${METRIC:-NOT SET}"

# Check for issues
echo ""
echo "3. Diagnosing issues:"
echo "   ─────────────────"

if [ "$MIN_SCALE" = "$MAX_SCALE" ] && [ -n "$MIN_SCALE" ]; then
    echo "   ❌ PROBLEM: min-scale == max-scale ($MIN_SCALE)"
    echo "      Scaling is DISABLED!"
    echo "      Fix: Set different values (e.g., min=0, max=3)"
fi

if [ -z "$MIN_SCALE" ]; then
    echo "   ⚠️  min-scale not set (defaults to 0)"
fi

if [ -z "$MAX_SCALE" ]; then
    echo "   ⚠️  max-scale not set (defaults to 0 = unlimited)"
fi

if [ -n "$TARGET" ] && [ "$TARGET" -gt 200 ]; then
    echo "   ⚠️  target is very high ($TARGET)"
    echo "      Knative won't scale until $TARGET concurrent requests"
    echo "      Consider: 50-100 for typical workloads"
fi

if [ -n "$TARGET" ] && [ "$TARGET" -lt 10 ]; then
    echo "   ⚠️  target is very low ($TARGET)"
    echo "      Knative may scale too aggressively"
fi

# Check current pod count
echo ""
echo "4. Current pods:"
echo "   ────────────"

PODS=$(kubectl get pods -l serving.knative.dev/service=$SERVICE_NAME --no-headers 2>/dev/null)
if [ -z "$PODS" ]; then
    echo "   No pods running (scaled to zero)"
else
    echo "$PODS" | awk '{print "   " $1 " - " $3}'
fi

POD_COUNT=$(echo "$PODS" | wc -l)
if [ -z "$PODS" ]; then
    POD_COUNT=0
fi

echo ""
echo "   Total pods: $POD_COUNT"

if [ "$POD_COUNT" -eq 1 ] && [ "$MIN_SCALE" = "1" ] && [ "$MAX_SCALE" = "1" ]; then
    echo "   ❌ Stuck at 1 pod because min-scale=max-scale=1"
fi

# Check recent events
echo ""
echo "5. Recent scaling events:"
echo "   ─────────────────────"

kubectl get events --sort-by='.lastTimestamp' | grep -i "$SERVICE_NAME\|autoscal" | tail -5 | awk '{print "   " $0}'

# Show recommendations
echo ""
echo "========================================"
echo "RECOMMENDATIONS"
echo "========================================"

if [ "$MIN_SCALE" = "$MAX_SCALE" ]; then
    echo ""
    echo "🔧 FIX: Enable scaling by setting different min/max:"
    echo ""
    echo "kubectl patch ksvc $SERVICE_NAME --type merge -p '{"
    echo '  "spec": {'
    echo '    "template": {'
    echo '      "metadata": {'
    echo '        "annotations": {'
    echo '          "autoscaling.knative.dev/min-scale": "1",'
    echo '          "autoscaling.knative.dev/max-scale": "3",'
    echo '          "autoscaling.knative.dev/target": "80"'
    echo '        }'
    echo '      }'
    echo '    }'
    echo '  }'
    echo "}'"
fi

echo ""
echo "🧪 TEST: Verify autoscaler can change min-scale:"
echo ""
echo "# Try manually changing min-scale"
echo "kubectl patch ksvc $SERVICE_NAME --type merge -p '{"
echo '  "spec": {'
echo '    "template": {'
echo '      "metadata": {'
echo '        "annotations": {'
echo '          "autoscaling.knative.dev/min-scale": "2"'
echo '        }'
echo '      }'
echo '    }'
echo '  }'
echo "}'"
echo ""
echo "# Then check if pods scale"
echo "watch kubectl get pods -l serving.knative.dev/service=$SERVICE_NAME"

echo ""
echo "========================================"