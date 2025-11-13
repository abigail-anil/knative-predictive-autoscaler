from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import h5py
from tensorflow.keras.models import model_from_json
from datetime import datetime, timedelta

app = FastAPI()

def fix_prophet_dates(model):
    """Ensure Prophet model history uses valid datetime format."""
    try:
        # Convert to datetime if not already
        if model.history['ds'].dtype != 'datetime64[ns]':
            model.history['ds'] = pd.to_datetime(model.history['ds'], errors='coerce')
        
        # Check if dates are in 1970 (epoch issue)
        min_year = model.history['ds'].dt.year.min()
        max_year = model.history['ds'].dt.year.max()
        
        if min_year == 1970 and max_year == 1970:
            print("[CRITICAL] Prophet model has 1970 timestamps!")
            print("[INFO] Attempting emergency fix...")
            
            # Emergency fix: reconstruct dates
            n_points = len(model.history)
            # Use January 2021 as baseline (matching your dataset)
            start_date = pd.Timestamp('2021-01-01 00:00:00')
            proper_dates = pd.date_range(start=start_date, periods=n_points, freq='1min')
            model.history['ds'] = proper_dates
            
            print(f"[FIXED] Reconstructed dates: {proper_dates[0]} to {proper_dates[-1]}")
        else:
            print(f"[OK] Prophet dates are valid: {min_year} to {max_year}")
            
    except Exception as e:
        print(f"[ERROR] Could not fix Prophet dates: {e}")
    
    return model


@app.get("/")
def root():
    return {"status": "ok", "message": "ML autoscaler service running"}

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "forecasting-api"}

MODEL_DIR = "/app/models"

# Lazy-loaded dictionaries
prophet_models = {}
lstm_models = {}
hybrid_models = {}

def safe_load_model(path):
    """Load an H5 Keras model safely even if compile metadata is broken."""
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        print(f"[WARN] Standard load_model failed for {path}: {e}")
        print("[INFO] Attempting manual load (architecture + weights only)")
        with h5py.File(path, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise RuntimeError(f"No model_config found in {path}")
            model = model_from_json(model_config.decode('utf-8'))
            weights_group = f['model_weights']
            for layer in model.layers:
                if layer.name in weights_group:
                    layer_weights = [
                        weights_group[layer.name][w_name][()]
                        for w_name in weights_group[layer.name]
                    ]
                    layer.set_weights(layer_weights)
        return model


class PredictRequest(BaseModel):
    periods: int = 5
    recent_data: list


def load_prophet(fid):
    if fid not in prophet_models:
        fname = f"{fid}_prophet.pkl"
        path = os.path.join(MODEL_DIR, "prophet", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prophet model not found: {path}")
        model = joblib.load(path)
        model = fix_prophet_dates(model)
        prophet_models[fid] = model
    return prophet_models[fid]


def load_lstm(fid):
    if fid not in lstm_models:
        fname = f"{fid}_lstm.h5"
        path = os.path.join(MODEL_DIR, "lstm", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LSTM model not found: {path}")
        lstm_models[fid] = safe_load_model(path)
    return lstm_models[fid]


def load_hybrid(fid):
    if fid not in hybrid_models:
        lstm_fname = f"{fid}_lstm_residual.h5"
        prophet_fname = f"{fid}_prophet.pkl"
        lstm_path = os.path.join(MODEL_DIR, "hybrid", lstm_fname)
        prophet_path = os.path.join(MODEL_DIR, "hybrid", prophet_fname)
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"Hybrid LSTM model not found: {lstm_path}")
        if not os.path.exists(prophet_path):
            raise FileNotFoundError(f"Hybrid Prophet model not found: {prophet_path}")
        
        prophet_model = joblib.load(prophet_path)
        prophet_model = fix_prophet_dates(prophet_model)
        
        hybrid_models[fid] = {
            "lstm": safe_load_model(lstm_path),
            "prophet": prophet_model
        }
    return hybrid_models[fid]


@app.post("/predict/{model_type}/{func_id}")
def predict_with_params(model_type: str, func_id: str, request: PredictRequest):
    """Original endpoint with model_type and func_id in URL"""
    return predict_internal(model_type, func_id, request)


@app.post("/predict")
def predict_simple(request: PredictRequest):
    """Simplified endpoint that reads model_type and func_id from environment"""
    model_type = os.getenv("MODEL_TYPE", "prophet").lower()
    func_id = os.getenv("FUNCTION_ID", "func_235")
    return predict_internal(model_type, func_id, request)


def predict_internal(model_type: str, func_id: str, request: PredictRequest):
    """Core prediction logic - FIXED VERSION"""
    fid = func_id
    model_type = model_type.lower()
    recent_data = request.recent_data
    periods = request.periods

    if not recent_data or not isinstance(recent_data, list):
        raise HTTPException(status_code=400, detail="recent_data must be a non-empty list")

    if model_type == "prophet":
        model = load_prophet(fid)
        
        # Get the last date from model history
        history_dates = model.history['ds']
        last_training_date = history_dates.max()
        
        print(f"[INFO] Prophet last training date: {last_training_date}")
        print(f"[INFO] Prophet date year: {last_training_date.year}")
        
        # CRITICAL FIX: Check if we have a valid date
        if last_training_date.year == 1970:
            print("[ERROR] Model still has 1970 dates after fix attempt!")
            # Use current time as fallback
            last_training_date = pd.Timestamp.now()
            print(f"[FALLBACK] Using current time: {last_training_date}")
        
        # Create future dates relative to training data
        # Use the inferred frequency from the model's history
        try:
            freq = pd.infer_freq(history_dates[-100:])  # Infer from last 100 points
            if freq is None:
                freq = '1min'  # Default to 1 minute
                print(f"[INFO] Could not infer frequency, using default: {freq}")
            else:
                print(f"[INFO] Inferred frequency: {freq}")
        except:
            freq = '1min'
            print(f"[INFO] Using default frequency: {freq}")
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_training_date + pd.Timedelta(minutes=1),
            periods=periods,
            freq=freq
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        print(f"[INFO] Predicting for dates: {future_dates[0]} to {future_dates[-1]}")
        
        # Make predictions
        forecast = model.predict(future_df)
        pred = forecast['yhat'].tolist()
        
        # Ensure non-negative predictions
        pred = [max(0, p) for p in pred]
        
        print(f"[INFO] Prophet raw predictions: {[f'{p:.1f}' for p in pred]}")
        
        # Check if predictions are suspiciously low
        if max(pred) < 1.0:
            print("[WARNING] Predictions are very low - model may need retraining")

    elif model_type == "lstm":
        model = load_lstm(fid)
        sequence_length = model.input_shape[1] if hasattr(model, 'input_shape') else 30
        x = np.array(recent_data[-sequence_length:]).reshape(1, -1, 1)
        
        predictions = []
        current_sequence = x.copy()
        
        for _ in range(periods):
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(float(next_pred))
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        pred = predictions

    elif model_type == "hybrid":
        models = load_hybrid(fid)
        
        # LSTM prediction
        sequence_length = models["lstm"].input_shape[1] if hasattr(models["lstm"], 'input_shape') else 30
        x = np.array(recent_data[-sequence_length:]).reshape(1, -1, 1)
        
        lstm_predictions = []
        current_sequence = x.copy()
        
        for _ in range(periods):
            next_pred = models["lstm"].predict(current_sequence, verbose=0)[0, 0]
            lstm_predictions.append(float(next_pred))
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        # Prophet prediction
        prophet_model = models["prophet"]
        last_date = prophet_model.history['ds'].max()
        
        # Check for 1970 issue
        if last_date.year == 1970:
            last_date = pd.Timestamp.now()
        
        freq = pd.infer_freq(prophet_model.history['ds'][-100:]) or '1min'
        future_df = pd.DataFrame({
            'ds': pd.date_range(start=last_date + pd.Timedelta(minutes=1), 
                               periods=periods, 
                               freq=freq)
        })
        
        forecast = prophet_model.predict(future_df)
        prophet_predictions = forecast['yhat'].tolist()
        
        # Combine predictions (simple average)
        pred = [(l + p) / 2 for l, p in zip(lstm_predictions, prophet_predictions)]

    elif model_type == "reactive":
        # Simple reactive baseline (no ML model)
        # Just use the latest observed value or a moving average
        if len(recent_data) == 0:
            raise HTTPException(status_code=400, detail="No recent_data provided")


        window = min(len(recent_data), 10)
        moving_avg = np.mean(recent_data[-window:])
        pred = [float(moving_avg)] * periods

        print(f"[INFO] Reactive baseline used: moving_avg={moving_avg:.2f}")

    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use: prophet, lstm, hybrid, or reactive")

    return {"predictions": pred, "model": model_type, "function": func_id}

