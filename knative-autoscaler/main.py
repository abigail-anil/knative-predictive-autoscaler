from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import os
import h5py
from tensorflow.keras.models import model_from_json

app = FastAPI()
MODEL_DIR = "/app/models"

# Lazy-loaded dictionaries
prophet_models = {}
lstm_models = {}
hybrid_models = {}

# SAFE MODEL LOADER 
def safe_load_model(path):
    #Load an H5 Keras model safely even if compile metadata is broken.
    
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
    func_id: str
    model_type: str  # prophet, lstm, hybrid
    recent_traffic: list


# ---------------- MODEL LOADERS 

def load_prophet(fid):
    if fid not in prophet_models:
        fname = f"func_{fid}_prophet.pkl"
        path = os.path.join(MODEL_DIR, "prophet", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prophet model not found: {path}")
        prophet_models[fid] = joblib.load(path)
    return prophet_models[fid]


def load_lstm(fid):
    if fid not in lstm_models:
        fname = f"func_{fid}_lstm.h5"
        path = os.path.join(MODEL_DIR, "lstm", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LSTM model not found: {path}")
        lstm_models[fid] = safe_load_model(path)
    return lstm_models[fid]


def load_hybrid(fid):
    if fid not in hybrid_models:
        lstm_fname = f"func_{fid}_lstm_residual.h5"
        prophet_fname = f"func_{fid}_prophet.pkl"
        lstm_path = os.path.join(MODEL_DIR, "hybrid", lstm_fname)
        prophet_path = os.path.join(MODEL_DIR, "hybrid", prophet_fname)
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"Hybrid LSTM model not found: {lstm_path}")
        if not os.path.exists(prophet_path):
            raise FileNotFoundError(f"Hybrid Prophet model not found: {prophet_path}")
        hybrid_models[fid] = {
            "lstm": safe_load_model(lstm_path),
            "prophet": joblib.load(prophet_path)
        }
    return hybrid_models[fid]



@app.post("/predict")
def predict(request: PredictRequest):
    fid = request.func_id
    model_type = request.model_type.lower()
    traffic = request.recent_traffic

    if not traffic or not isinstance(traffic, list):
        raise HTTPException(status_code=400, detail="recent_traffic must be a non-empty list")

    if model_type == "prophet":
        model = load_prophet(fid)
        pred = model.predict(traffic)

    elif model_type == "lstm":
        model = load_lstm(fid)
        x = np.array(traffic).reshape(1, -1, 1)
        pred = model.predict(x).flatten().tolist()

    elif model_type == "hybrid":
        models = load_hybrid(fid)
        x = np.array(traffic).reshape(1, -1, 1)
        lstm_pred = models["lstm"].predict(x).flatten()
        prophet_pred = models["prophet"].predict(traffic)
        pred = (lstm_pred + np.array(prophet_pred)).tolist()

    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use 'prophet', 'lstm', or 'hybrid'.")

    return {"prediction": pred}


@app.get("/")
def health_check():
    return {"status": "ok", "message": "ML autoscaler service running"}
