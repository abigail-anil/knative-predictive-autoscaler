#!/usr/bin/env python3
"""
Shift Prophet model dates to the present day
This allows the model to make predictions for current/future times
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def shift_prophet_to_present(model_path, output_path=None, days_ago=7):
    """
    Shift Prophet model training dates to recent past
    
    """
    print(f"\n{'='*80}")
    print(f"Processing: {model_path}")
    print(f"{'='*80}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Get current date info
    original_start = model.history['ds'].min()
    original_end = model.history['ds'].max()
    training_duration = original_end - original_start
    
    print(f"\n ORIGINAL DATES:")
    print(f"  Start: {original_start}")
    print(f"  End:   {original_end}")
    print(f"  Duration: {training_duration}")
    
    # Calculate new dates
    # Place the END of training data X days ago
    new_end = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
    new_start = new_end - training_duration
    
    print(f"\n NEW DATES:")
    print(f"  Start: {new_start}")
    print(f"  End:   {new_end}")
    print(f"  Duration: {training_duration} (preserved)")
    
    # Calculate the time shift
    time_shift = new_end - original_end
    print(f"\n Time shift: {time_shift.days} days forward")
    
    # Apply shift to all datetime columns
    model.history['ds'] = model.history['ds'] + time_shift
    
    # Verify
    print(f"\n VERIFICATION:")
    print(f"  New date range: {model.history['ds'].min()} to {model.history['ds'].max()}")
    print(f"  Years: {model.history['ds'].dt.year.min()} to {model.history['ds'].dt.year.max()}")
    
    # Test prediction
    print(f"\n TESTING PREDICTIONS:")
    last_date = model.history['ds'].max()
    future = pd.DataFrame({
        'ds': pd.date_range(start=last_date + pd.Timedelta(minutes=1), 
                           periods=5, 
                           freq='1min')
    })
    
    print(f"  Predicting for: {future['ds'].iloc[0]} to {future['ds'].iloc[-1]}")
    
    forecast = model.predict(future)
    predictions = forecast['yhat'].values
    
    print(f"  Predictions: {predictions}")
    print(f"  Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")
    
    # Check if predictions are reasonable
    if predictions.min() < -1000:
        print(f"  WARNING: Very negative predictions detected!")
    elif predictions.max() < 0:
        print(f"  WARNING: All predictions are negative!")
    elif all(p == 0 for p in predictions):
        print(f"  WARNING: All predictions are zero!")
    else:
        print(f" Predictions look reasonable!")
    
    # Save
    if output_path is None:
        output_path = model_path
    
    joblib.dump(model, output_path)
    print(f"\n Saved to: {output_path}")
    
    return model


def shift_all_prophet_models(models_dir='models', days_ago=7):
    """
    Shift all Prophet models to present day
    """
    print("\n" + "="*80)
    print("SHIFT PROPHET MODELS TO PRESENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Models directory: {models_dir}")
    print(f"  Training end date: {days_ago} days ago")
    print(f"  Target date: {pd.Timestamp.now() - pd.Timedelta(days=days_ago)}")
    
    # Process standalone Prophet models
    prophet_dir = Path(models_dir) / 'prophet'
    if prophet_dir.exists():
        print(f"\nProcessing Prophet models in: {prophet_dir}")
        for model_file in prophet_dir.glob('*_prophet.pkl'):
            try:
                shift_prophet_to_present(model_file, days_ago=days_ago)
            except Exception as e:
                print(f" Error: {e}")
    
    # Process Hybrid Prophet models
    hybrid_dir = Path(models_dir) / 'hybrid'
    if hybrid_dir.exists():
        print(f"\nProcessing Hybrid Prophet models in: {hybrid_dir}")
        for model_file in hybrid_dir.glob('*_prophet.pkl'):
            try:
                shift_prophet_to_present(model_file, days_ago=days_ago)
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\n" + "="*80)
    print("ALL MODELS SHIFTED")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Shift Prophet model dates to present day',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        
    )
    
    parser.add_argument('--models-dir', default='models', 
                       help='Path to models directory')
    parser.add_argument('--days-ago', type=int, default=7,
                       help='Place training data end this many days ago (default: 7)')
    parser.add_argument('--single', help='Process single model file instead of all')
    
    args = parser.parse_args()
    
    if args.single:
        shift_prophet_to_present(args.single, days_ago=args.days_ago)
    else:
        shift_all_prophet_models(args.models_dir, args.days_ago)
    
 