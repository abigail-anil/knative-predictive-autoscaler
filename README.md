**Predictive Autoscaling for Knative Using Machine Learning**

This project brings together forecasting models and Knative autoscaling to see how far proactive scaling can reduce cold-start delays in serverless workloads. It uses real cloud invocation data, trains three prediction models, and plugs their results into a custom autoscaler running on Kubernetes.

**What this project does**

The core idea is straightforward: instead of waiting for traffic to spike, predict what’s coming and scale early. The codebase includes everything needed to train the models, deploy them as forecasting microservices, and integrate their predictions with Knative.

Here’s what’s inside:

A preprocessing and training pipeline for Prophet, LSTM and Hybrid (Prophet residuals + LSTM) models

* A FastAPI service for serving predictions in real time

* A modified Knative autoscaler extension that consumes those predictions

* Kubernetes/Knative deployment files for running the full setup on AKS

* Load-testing scripts and helper utilities for capturing traffic behaviour

**Why this matters**

Knative’s default autoscaler is reactive. It waits for concurrency to rise before scaling out. That delay creates cold-start penalties during bursts.
This project explores whether machine-learning forecasts can soften that lag by triggering pods a little earlier.

The goal isn’t to replace reactive scaling but to extend it for scenarios where anticipation actually helps.
