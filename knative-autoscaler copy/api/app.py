"""
Unified Forecasting API for Knative Deployment
Serves Prophet, LSTM, and Hybrid models
"""
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
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

    
FUNCTION_ID = os.environ.get('FUNCTION_ID', 'func_235')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'prophet')

def load_models():
    """Load only the required model based on environment variables"""
    logger.info(f"Loading {MODEL_TYPE} model for {FUNCTION_ID}...")
    
    try:
        if MODEL_TYPE == 'prophet':
            prophet_path = f"{MODEL_DIR}/prophet/{FUNCTION_ID}_prophet.pkl"
            if os.path.exists(prophet_path):
                MODELS['prophet'][FUNCTION_ID] = joblib.load(prophet_path)
                logger.info(f"Loaded Prophet for {FUNCTION_ID}")
            else:
                logger.error(f"Prophet model not found: {prophet_path}")
        
        elif MODEL_TYPE == 'lstm':
            lstm_model_path = f"{MODEL_DIR}/lstm/{FUNCTION_ID}_lstm.h5"
            lstm_scalers_path = f"{MODEL_DIR}/lstm/{FUNCTION_ID}_scalers.pkl"
            if os.path.exists(lstm_model_path) and os.path.exists(lstm_scalers_path):
                lstm_model = tf.keras.models.load_model(lstm_model_path, compile=False)
                scalers = joblib.load(lstm_scalers_path)
                MODELS['lstm'][FUNCTION_ID] = {
                    'model': lstm_model,
                    'X_scaler': scalers['X_scaler'],
                    'y_scaler': scalers['y_scaler'],
                    'lookback': scalers['lookback']
                }
                logger.info(f"Loaded LSTM for {FUNCTION_ID}")
            else:
                logger.error(f"LSTM model not found: {lstm_model_path}")
        
        elif MODEL_TYPE == 'hybrid':
            hybrid_prophet_path = f"{MODEL_DIR}/hybrid/{FUNCTION_ID}_prophet.pkl"
            hybrid_lstm_path = f"{MODEL_DIR}/hybrid/{FUNCTION_ID}_lstm_residual.h5"
            hybrid_scalers_path = f"{MODEL_DIR}/hybrid/{FUNCTION_ID}_scalers.pkl"
            
            if all(os.path.exists(p) for p in [hybrid_prophet_path, hybrid_lstm_path, hybrid_scalers_path]):
                prophet_model = joblib.load(hybrid_prophet_path)
                lstm_model = tf.keras.models.load_model(hybrid_lstm_path, compile=False)
                scalers = joblib.load(hybrid_scalers_path)
                
                MODELS['hybrid'][FUNCTION_ID] = {
                    'prophet_model': prophet_model,
                    'lstm_model': lstm_model,
                    'X_scaler': scalers['X_scaler'],
                    'y_scaler': scalers['y_scaler'],
                    'lookback': scalers['lookback']
                }
                logger.info(f"Loaded Hybrid for {FUNCTION_ID}")
            else:
                logger.error(f"Hybrid model not found")
        
        elif MODEL_TYPE == 'reactive':
            # Reactive doesn't need models
            logger.info("Reactive mode - no models needed")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()

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
        "recent_data": [100, 105, 110, ...],  
        "current_time": "2024-01-15T10:30:00" 
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
        current_time = data.get('current_time', None)
        
        # Generate predictions based on model type
        if model_type == 'prophet':
            predictions = predict_prophet(function_id, periods, recent_data, current_time)
            
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def predict_prophet(function_id, periods, recent_data=None, current_time=None):
    """
    Generate Prophet predictions
    
    Prophet needs historical context. If recent_data is provided, we feed it to Prophet.
    Otherwise, we just predict from the last training point.
    """
    model = MODELS['prophet'][function_id]
    
    from datetime import datetime, timedelta
    
    # Determine starting point
    if current_time:
        start_time = pd.to_datetime(current_time)
    else:
        start_time = datetime.now()
    
    # If recent data is provided, create a mini-history for better predictions
    if recent_data and len(recent_data) > 0:
        # Create timestamps for recent data (going backwards from current_time)
        timestamps = [start_time - timedelta(minutes=len(recent_data)-i) 
                     for i in range(len(recent_data))]
        
        # Create dataframe with recent history
        history = pd.DataFrame({
            'ds': timestamps,
            'y': recent_data
        })
        
       
    # Create future dataframe
    future_dates = [start_time + timedelta(minutes=i+1) for i in range(periods)]
    future = pd.DataFrame({'ds': future_dates})
    
    try:
        # Predict
        forecast = model.predict(future)
        predictions = forecast['yhat'].values
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        logger.info(f"Prophet predictions for {function_id}: {predictions[:3]}... (showing first 3)")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prophet prediction error: {e}")
        # Fallback: return mean of recent data if available
        if recent_data and len(recent_data) > 0:
            mean_value = np.mean(recent_data)
            return np.full(periods, mean_value)
        else:
            return np.zeros(periods)

def predict_lstm(function_id, recent_data, periods):
    #Generate LSTM predictions
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
    #Generate Hybrid predictions (Prophet + LSTM on residuals)
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
    #List all available models
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