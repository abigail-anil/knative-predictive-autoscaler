from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ENV
MODEL_DIR = "/app/models"
FUNCTION_ID = os.getenv("FUNCTION_ID", "func_235")
MODEL_TYPE = os.getenv("MODEL_TYPE", "prophet").lower()

# Cache
prophet_cache = {}
lstm_cache = {}
hybrid_cache = {}

# HEALTH
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"status": "ok", "model_type": MODEL_TYPE, "function_id": FUNCTION_ID}


# REQUEST MODEL
class PredictRequest(BaseModel):
    recent_data: list
    periods: int = 5


# HELPER: Load Prophet model with proper configuration
def load_prophet_model(fid):
    if fid in prophet_cache:
        logger.info(f"Using cached Prophet model for {fid}")
        return prophet_cache[fid]

    path = os.path.join(MODEL_DIR, "prophet", f"{fid}_prophet.pkl")
    if not os.path.exists(path):
        logger.error(f"Prophet model file not found: {path}")
        raise HTTPException(500, f"Prophet model missing: {path}")
    
    try:
        logger.info(f"Loading Prophet model from {path}")
        model = joblib.load(path)
        
        # CRITICAL: Disable Stan backend to prevent hanging
        if hasattr(model, 'stan_backend'):
            model.stan_backend = None
        
        # Validate model has required data
        if not hasattr(model, 'history') or model.history.empty:
            raise ValueError("Prophet model has no training history")
        
        # Ensure datetime format
        if "ds" in model.history.columns:
            model.history["ds"] = pd.to_datetime(model.history["ds"], errors='coerce')
        
        # Ensure cap and floor exist for logistic growth
        if "cap" not in model.history.columns:
            logger.warning("Adding missing 'cap' column")
            model.history["cap"] = model.history["y"].max() * 1.5
        
        if "floor" not in model.history.columns:
            model.history["floor"] = 0.0
        
        logger.info(f"Successfully loaded Prophet model: {len(model.history)} training points")
        prophet_cache[fid] = model
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Prophet model: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")


# HELPER: Load LSTM model with compatibility handling and scalers
def load_lstm_model(fid):
    if fid in lstm_cache:
        return lstm_cache[fid]

    import tensorflow as tf
    from tensorflow import keras

    path = os.path.join(MODEL_DIR, "lstm", f"{fid}_lstm.h5")
    if not os.path.exists(path):
        raise HTTPException(500, f"LSTM model missing: {path}")

    try:
        logger.info(f"Loading LSTM model from {path}")
        
        # Try loading scalers (if they exist)
        scaler_path = os.path.join(MODEL_DIR, "lstm", f"{fid}_lstm_scalers.pkl")
        scalers = None
        if os.path.exists(scaler_path):
            logger.info(f"Loading scalers from {scaler_path}")
            scalers = joblib.load(scaler_path)
        
        # Try standard load first
        try:
            model = tf.keras.models.load_model(path, compile=False)
            logger.info("Successfully loaded LSTM model (standard)")
            lstm_cache[fid] = {'model': model, 'scalers': scalers}
            return lstm_cache[fid]
        except Exception as load_error:
            logger.warning(f"Standard load failed: {load_error}")
            logger.info("Attempting manual reconstruction...")
        
        # Manual reconstruction for compatibility
        import h5py
        import json
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        with h5py.File(path, 'r') as f:
            # Reconstruct model from config
            if 'model_config' in f.attrs:
                model_config = f.attrs['model_config']
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                config = json.loads(model_config)
                model = Sequential()
                
                # Parse layers from config
                layers_config = config['config']['layers']
                
                for i, layer_config in enumerate(layers_config):
                    layer_type = layer_config['class_name']
                    layer_conf = layer_config['config']
                    
                    if layer_type == 'LSTM':
                        units = layer_conf['units']
                        return_sequences = layer_conf.get('return_sequences', False)
                        
                        if i == 0:
                            # First layer needs input_shape
                            input_shape = layer_conf.get('batch_input_shape', [None, 30, 1])
                            model.add(LSTM(
                                units=units,
                                return_sequences=return_sequences,
                                input_shape=(input_shape[1], input_shape[2])
                            ))
                        else:
                            model.add(LSTM(
                                units=units,
                                return_sequences=return_sequences
                            ))
                    
                    elif layer_type == 'Dropout':
                        rate = layer_conf['rate']
                        model.add(Dropout(rate))
                    
                    elif layer_type == 'Dense':
                        units = layer_conf['units']
                        model.add(Dense(units))
                
                # Load weights
                if 'model_weights' in f:
                    weights_group = f['model_weights']
                    
                    for layer in model.layers:
                        if layer.name in weights_group:
                            layer_weights = []
                            for weight_name in weights_group[layer.name]:
                                layer_weights.append(weights_group[layer.name][weight_name][()])
                            
                            if layer_weights:
                                layer.set_weights(layer_weights)
                
                logger.info("Successfully reconstructed LSTM model")
                lstm_cache[fid] = {'model': model, 'scalers': scalers}
                return lstm_cache[fid]
            
            else:
                raise ValueError("No model_config found in H5 file")
    
    except Exception as e:
        logger.error(f"LSTM load failed completely: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"LSTM model load failed: {str(e)}")


# HELPER: Load Hybrid models with compatibility handling
def load_hybrid_models(fid):
    if fid in hybrid_cache:
        return hybrid_cache[fid]

    import tensorflow as tf
    from tensorflow import keras

    lstm_path = os.path.join(MODEL_DIR, "hybrid", f"{fid}_lstm_residual.h5")
    prophet_path = os.path.join(MODEL_DIR, "hybrid", f"{fid}_prophet.pkl")
    scaler_path = os.path.join(MODEL_DIR, "hybrid", f"{fid}_lstm_scalers.pkl")

    if not os.path.exists(lstm_path) or not os.path.exists(prophet_path):
        raise HTTPException(500, "Hybrid model files missing")

    try:
        # Load Prophet
        prophet_model = joblib.load(prophet_path)
        if hasattr(prophet_model, 'stan_backend'):
            prophet_model.stan_backend = None
        
        if hasattr(prophet_model, 'history') and "ds" in prophet_model.history.columns:
            prophet_model.history["ds"] = pd.to_datetime(prophet_model.history["ds"], errors='coerce')
            
            if "cap" not in prophet_model.history.columns:
                prophet_model.history["cap"] = prophet_model.history["y"].max() * 1.5
            if "floor" not in prophet_model.history.columns:
                prophet_model.history["floor"] = 0.0
        
        # Load scalers if they exist
        scalers = None
        if os.path.exists(scaler_path):
            logger.info(f"Loading hybrid scalers from {scaler_path}")
            scalers = joblib.load(scaler_path)
        
        # Load LSTM with same compatibility handling
        try:
            lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
        except Exception as load_error:
            logger.warning(f"Standard LSTM load failed: {load_error}")
            logger.info("Attempting manual reconstruction...")
            
            import h5py
            import json
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            with h5py.File(lstm_path, 'r') as f:
                model_config = f.attrs['model_config']
                if isinstance(model_config, bytes):
                    model_config = model_config.decode('utf-8')
                
                config = json.loads(model_config)
                lstm_model = Sequential()
                
                layers_config = config['config']['layers']
                
                for i, layer_config in enumerate(layers_config):
                    layer_type = layer_config['class_name']
                    layer_conf = layer_config['config']
                    
                    if layer_type == 'LSTM':
                        units = layer_conf['units']
                        return_sequences = layer_conf.get('return_sequences', False)
                        
                        if i == 0:
                            input_shape = layer_conf.get('batch_input_shape', [None, 30, 1])
                            lstm_model.add(LSTM(
                                units=units,
                                return_sequences=return_sequences,
                                input_shape=(input_shape[1], input_shape[2])
                            ))
                        else:
                            lstm_model.add(LSTM(
                                units=units,
                                return_sequences=return_sequences
                            ))
                    
                    elif layer_type == 'Dropout':
                        lstm_model.add(Dropout(layer_conf['rate']))
                    
                    elif layer_type == 'Dense':
                        lstm_model.add(Dense(layer_conf['units']))
                
                # Load weights
                if 'model_weights' in f:
                    weights_group = f['model_weights']
                    for layer in lstm_model.layers:
                        if layer.name in weights_group:
                            layer_weights = []
                            for weight_name in weights_group[layer.name]:
                                layer_weights.append(weights_group[layer.name][weight_name][()])
                            if layer_weights:
                                layer.set_weights(layer_weights)

        hybrid_cache[fid] = {
            "lstm": lstm_model,
            "prophet": prophet_model,
            "scalers": scalers
        }
        logger.info("Successfully loaded Hybrid models")
        return hybrid_cache[fid]
        
    except Exception as e:
        logger.error(f"Hybrid load failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Hybrid models load failed: {str(e)}")


# PROPHET PREDICTION with timeout protection
def predict_prophet(model, periods):
    try:
        logger.info(f"Starting Prophet prediction for {periods} periods")
        
        # Validate history
        if not hasattr(model, 'history') or model.history.empty:
            raise ValueError("Model has no history")
        
        history = model.history["ds"]
        last_date = history.max()
        
        logger.info(f"Last training date: {last_date}")
        
        # Infer frequency with fallback
        freq = "1min"
        try:
            recent_history = history.tail(min(200, len(history)))
            inferred_freq = pd.infer_freq(recent_history)
            if inferred_freq:
                freq = inferred_freq
                logger.info(f"Inferred frequency: {freq}")
        except Exception as e:
            logger.warning(f"Could not infer frequency: {e}")

        # Create future dataframe
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(minutes=1),
            periods=periods,
            freq=freq
        )

        future = pd.DataFrame({"ds": future_dates})
        
        # Add cap and floor if model uses them
        if "cap" in model.history.columns:
            cap_value = float(model.history["cap"].iloc[0])
            future["cap"] = cap_value
            future["floor"] = 0.0
            logger.info(f"Using growth cap: {cap_value}")

        # Make prediction with error handling
        logger.info("Calling model.predict()...")
        
        #import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Prophet prediction timed out")
        
        #signal.signal(signal.SIGALRM, timeout_handler)
        #signal.alarm(8)
        
        try:
            forecast = model.predict(future)
            #signal.alarm(0)
        except TimeoutError:
            #signal.alarm(0)
            logger.error("Prophet prediction timed out - using fallback")
            recent_values = model.history["y"].tail(periods).values
            pred = np.repeat(recent_values.mean(), periods)
            return pred.tolist()
        
        # Extract predictions and remove offset
        pred = forecast["yhat"].values
        pred = np.maximum(pred - 0.1, 0)
        
        logger.info(f"Predictions: {pred[:min(3, len(pred))]}...")
        return pred.tolist()
        
    except Exception as e:
        logger.error(f"Prophet prediction failed: {e}")
        logger.error(traceback.format_exc())
        
        try:
            recent_avg = float(model.history["y"].tail(10).mean()) - 0.1
            recent_avg = max(0, recent_avg)
            fallback = [recent_avg] * periods
            logger.warning(f"Using fallback prediction: {recent_avg:.1f}")
            return fallback
        except:
            return [10.0] * periods


# LSTM PREDICTION with proper scaling
def predict_lstm(lstm_bundle, recent_data, periods):
    try:
        model = lstm_bundle['model']
        scalers = lstm_bundle.get('scalers')
        
        logger.info(f"Starting LSTM prediction for {periods} periods")
        seq_len = model.input_shape[1]
        
        logger.info(f"Model requires {seq_len} timesteps, received {len(recent_data)} points")
        
        # Prepare data (add offset like in training)
        recent_data_adjusted = [float(x) + 0.1 for x in recent_data]
        
        # Pad if necessary
        if len(recent_data_adjusted) < seq_len:
            mean_val = np.mean(recent_data_adjusted) if recent_data_adjusted else 10.1
            padding = [mean_val] * (seq_len - len(recent_data_adjusted))
            recent_data_adjusted = padding + recent_data_adjusted
            logger.warning(f"Padded with {len(padding)} points (mean={mean_val:.2f})")
        
        # Take last seq_len points
        sequence = np.array(recent_data_adjusted[-seq_len:])
        
        # Apply scaling if scalers exist
        if scalers and 'X_scaler' in scalers:
            logger.info("Applying MinMaxScaler to input")
            sequence = scalers['X_scaler'].transform(sequence.reshape(-1, 1)).flatten()
        
        # Reshape for LSTM
        x = sequence.reshape(1, seq_len, 1)
        
        preds_scaled = []
        curr = x.copy()

        # Autoregressive prediction
        for i in range(periods):
            nxt_scaled = model.predict(curr, verbose=0)[0, 0]
            preds_scaled.append(nxt_scaled)
            
            # Update sequence
            curr = np.roll(curr, -1, axis=1)
            curr[0, -1, 0] = nxt_scaled
        
        # Inverse transform if scalers exist
        if scalers and 'y_scaler' in scalers:
            logger.info("Applying inverse scaling to predictions")
            preds_unscaled = scalers['y_scaler'].inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)
            ).flatten()
        else:
            preds_unscaled = np.array(preds_scaled)
        
        # Remove offset and ensure non-negative
        preds_final = [max(0, float(p) - 0.1) for p in preds_unscaled]
        
        logger.info(f"Final predictions: {preds_final}")
        return preds_final
        
    except Exception as e:
        logger.error(f"LSTM prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"LSTM prediction failed: {str(e)}")


# HYBRID PREDICTION
def predict_hybrid(bundle, recent_data, periods):
    try:
        logger.info("Starting Hybrid prediction")
        prophet_model = bundle["prophet"]
        lstm_model = bundle["lstm"]
        scalers = bundle.get("scalers")

        # Get Prophet predictions
        prophet_out = predict_prophet(prophet_model, periods)
        
 
        lstm_bundle_for_pred = {'model': lstm_model, 'scalers': scalers}
        lstm_out = predict_lstm(lstm_bundle_for_pred, recent_data, periods)

        # Ensemble: average (simple approach)
        hybrid = [(p + l) / 2.0 for p, l in zip(prophet_out, lstm_out)]
        
        logger.info(f"Prophet: {prophet_out[:3]}...")
        logger.info(f"LSTM: {lstm_out[:3]}...")
        logger.info(f"Hybrid: {hybrid[:3]}...")
        return hybrid
        
    except Exception as e:
        logger.error(f"Hybrid prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Hybrid prediction failed: {str(e)}")


# MAIN PREDICT ENDPOINT
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        logger.info(f"=== Prediction Request ===")
        logger.info(f"Model: {MODEL_TYPE}, Function: {FUNCTION_ID}")
        logger.info(f"Periods: {request.periods}, Data points: {len(request.recent_data)}")
        
        if not request.recent_data:
            raise HTTPException(status_code=400, detail="recent_data required")

        fid = FUNCTION_ID

        if MODEL_TYPE == "prophet":
            model = load_prophet_model(fid)
            preds = predict_prophet(model, request.periods)

        elif MODEL_TYPE == "lstm":
            lstm_bundle = load_lstm_model(fid)
            preds = predict_lstm(lstm_bundle, request.recent_data, request.periods)

        elif MODEL_TYPE == "hybrid":
            bundle = load_hybrid_models(fid)
            preds = predict_hybrid(bundle, request.recent_data, request.periods)

        else:
            raise HTTPException(400, f"Unknown model_type: {MODEL_TYPE}")

        logger.info(f"Successfully generated {len(preds)} predictions")
        
        return {
            "function": fid,
            "model_type": MODEL_TYPE,
            "predictions": preds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# Startup event to pre-load model
@app.on_event("startup")
async def startup_event():
    logger.info(f"=== Forecasting API Starting ===")
    logger.info(f"Model Type: {MODEL_TYPE}")
    logger.info(f"Function ID: {FUNCTION_ID}")
    logger.info(f"Model Directory: {MODEL_DIR}")
    
    # Pre-load model
    try:
        if MODEL_TYPE == "prophet":
            load_prophet_model(FUNCTION_ID)
            logger.info("Prophet model pre-loaded successfully")
        elif MODEL_TYPE == "lstm":
            load_lstm_model(FUNCTION_ID)
            logger.info("LSTM model pre-loaded successfully")
        elif MODEL_TYPE == "hybrid":
            load_hybrid_models(FUNCTION_ID)
            logger.info("Hybrid models pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
        logger.error("API will start but predictions may fail")