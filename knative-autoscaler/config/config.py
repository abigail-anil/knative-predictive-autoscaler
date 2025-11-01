"""
Configuration for all autoscalers.
"""

import os

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PROPHET_MODEL_PATH = os.path.join(MODEL_DIR, 'prophet_model_func_235.pkl')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model_func_235.h5')
SCALERS_PATH = os.path.join(MODEL_DIR, 'scalers_func_235.pkl')

# Autoscaling parameters
PREDICTION_HORIZON = 60  # seconds
REQUESTS_PER_POD = 50    # Each pod can handle 50 req/min
SAFETY_BUFFER = 1.2      # 20% buffer
MIN_PODS = 1
MAX_PODS = 10
CHECK_INTERVAL = 60      # Check every 60 seconds

# Kubernetes
KNATIVE_NAMESPACE = "default"
KNATIVE_SERVICE_NAME = "hello-world"

# Model API
MODEL_API_HOST = "0.0.0.0"
MODEL_API_PORT = 5000

# Data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAFFIC_DATA_PATH = os.path.join(DATA_DIR, 'timeseries_func_235.csv')

# Results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')