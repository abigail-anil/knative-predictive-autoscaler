"""
Unified Forecasting API for Knative Deployment
Serves Prophet, LSTM, and Hybrid models
"""
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import time
import logging
from pathlib import Path

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.environ.get('MODEL_DIR', '/app/models')
LOOKBACK = 30  # Must match your LSTM_LOOKBACK

# Global model cache
MODELS = {
    'prophet': {},
    'lstm': {},
    'hybrid': {}
}

def load_models():
    """Load all pre-trained models into memory"""
    logger.info("Loading pre-trained models...")
    
    functions = ['func_235', 'func_242', 'func_126', 'func_186',
                 'func_190', 'func_110', 'func_99', 'func_191']
    
    for func_id in functions:
        try:
            # Load Prophet
            prophet_path = f"{MODEL_DIR}/prophet/{func_id}_prophet.pkl"
            if os.path.exists(prophet_path):
                MODELS['prophet'][func_id] = joblib.load(prophet_path)
                logger.info(f"  ✓ Loaded Prophet for {func_id}")
            
            # Load LSTM
            lstm_path = f"{MODEL_DIR}/lstm/{func_id}_lstm.pkl"
            if os.path.exists(lstm_path):
                MODELS['lstm'][func_id] = joblib.load(lstm_path)
                logger.info(f"  ✓ Loaded LSTM for {func_id}")
            
            # Load Hybrid
            hybrid_path = f"{MODEL_DIR}/hybrid/{func_id}_hybrid.pkl"
            if os.path.exists(hybrid_path):
                MODELS['hybrid'][func_id] = joblib.load(hybrid_path)
                logger.info(f"  ✓ Loaded Hybrid for {func_id}")
                
        except Exception as e:
            logger.error(f"  ✗ Error loading models for {func_id}: {str(e)}")
    
    logger.info(f"Models loaded: Prophet={len(MODELS['prophet'])}, "
                f"LSTM={len(MODELS['lstm'])}, Hybrid={len(MODELS['hybrid'])}")

# Load models on startup
load_models()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'prophet': len(MODELS['prophet']),
            'lstm': len(MODELS['lstm']),
            'hybrid': len(MODELS['hybrid'])
        },
        'timestamp': time.time()
    })

@app.route('/predict/<model_type>/<function_id>', methods=['POST'])
def predict(model_type, function_id):
    """
    Generate forecast for autoscaling
    
    POST /predict/prophet/func_235
    Body: {
        "periods": 12,
        "recent_data": [100, 105, 110, ...]  // Last 30+ data points (for LSTM/Hybrid)
    }
    
    Returns: {
        "function_id": "func_235",
        "model_type": "prophet",
        "predictions": [120, 125, 130, ...],
        "inference_time_ms": 45.2
    }
    """
    try:
        start_time = time.time()
        
        # Validate model type
        if model_type not in ['prophet', 'lstm', 'hybrid']:
            return jsonify({'error': f'Invalid model type: {model_type}'}), 400
        
        # Check if model exists
        if function_id not in MODELS[model_type]:
            return jsonify({'error': f'Model not found: {model_type}/{function_id}'}), 404
        
        # Parse request
        data = request.get_json()
        periods = data.get('periods', 12)  # Default 12 minutes ahead
        recent_data = data.get('recent_data', [])
        
        # Generate predictions based on model type
        if model_type == 'prophet':
            predictions = predict_prophet(function_id, periods)
            
        elif model_type == 'lstm':
            if len(recent_data) < LOOKBACK:
                return jsonify({
                    'error': f'LSTM requires at least {LOOKBACK} recent data points'
                }), 400
            predictions = predict_lstm(function_id, recent_data, periods)
            
        elif model_type == 'hybrid':
            if len(recent_data) < LOOKBACK:
                return jsonify({
                    'error': f'Hybrid requires at least {LOOKBACK} recent data points'
                }), 400
            predictions = predict_hybrid(function_id, recent_data, periods)
        
        inference_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'function_id': function_id,
            'model_type': model_type,
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'periods': periods,
            'inference_time_ms': round(inference_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def predict_prophet(function_id, periods):
    """Generate Prophet predictions"""
    model = MODELS['prophet'][function_id]
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='1min')
    
    # Predict
    forecast = model.predict(future)
    
    # Return only future predictions
    predictions = forecast['yhat'].tail(periods).values
    
    # Ensure non-negative
    predictions = np.maximum(predictions, 0)
    
    return predictions

def predict_lstm(function_id, recent_data, periods):
    """Generate LSTM predictions"""
    lstm_data = MODELS['lstm'][function_id]
    
    model = lstm_data['model']
    X_scaler = lstm_data['X_scaler']
    y_scaler = lstm_data['y_scaler']
    
    # Convert to numpy array
    recent_data = np.array(recent_data)
    
    # Take last LOOKBACK points
    current_sequence = recent_data[-LOOKBACK:]
    
    # Scale
    scaled_sequence = X_scaler.transform(current_sequence.reshape(-1, 1))
    
    predictions = []
    
    for _ in range(periods):
        # Reshape for prediction
        x_input = scaled_sequence[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        
        # Predict
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        
        # Update sequence
        scaled_sequence = np.append(scaled_sequence, [[pred_scaled]], axis=0)
    
    # Inverse transform
    predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predictions = np.maximum(predictions.flatten(), 0)
    
    return predictions

def predict_hybrid(function_id, recent_data, periods):
    """Generate Hybrid predictions (Prophet + LSTM on residuals)"""
    hybrid_data = MODELS['hybrid'][function_id]
    
    prophet_model = hybrid_data['prophet_model']
    lstm_model = hybrid_data['lstm_model']
    X_scaler = hybrid_data['X_scaler']
    y_scaler = hybrid_data['y_scaler']
    
    # 1. Prophet prediction
    future = prophet_model.make_future_dataframe(periods=periods, freq='1min')
    forecast = prophet_model.predict(future)
    prophet_pred = forecast['yhat'].tail(periods).values
    
    # 2. LSTM prediction on residuals
    recent_data = np.array(recent_data)
    current_sequence = recent_data[-LOOKBACK:]
    scaled_sequence = X_scaler.transform(current_sequence.reshape(-1, 1))
    
    lstm_corrections = []
    
    for _ in range(periods):
        x_input = scaled_sequence[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        pred_scaled = lstm_model.predict(x_input, verbose=0)[0, 0]
        lstm_corrections.append(pred_scaled)
        scaled_sequence = np.append(scaled_sequence, [[pred_scaled]], axis=0)
    
    lstm_corrections = y_scaler.inverse_transform(
        np.array(lstm_corrections).reshape(-1, 1)
    ).flatten()
    
    # 3. Combine predictions
    hybrid_pred = prophet_pred + lstm_corrections
    hybrid_pred = np.maximum(hybrid_pred, 0)
    
    return hybrid_pred

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    available = {}
    for model_type in ['prophet', 'lstm', 'hybrid']:
        available[model_type] = list(MODELS[model_type].keys())
    
    return jsonify({
        'available_models': available,
        'total': sum(len(v) for v in available.values())
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)