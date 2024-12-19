import pandas as pd
import numpy as np
import pickle

def load_preprocessing_pipeline(filepath):
    """
    Loads the saved preprocessing pipeline from disk.
    
    Parameters:
    - filepath (str): Path to the saved preprocessing pipeline.
    
    Returns:
    - pipeline (sklearn.pipeline.Pipeline): The loaded preprocessing pipeline.
    """
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def load_model(filepath):
    """
    Loads the trained ML model from disk.
    
    Parameters:
    - filepath (str): Path to the saved model file.
    
    Returns:
    - model (RandomForestRegressor): The loaded ML model.
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_new_entry(new_entry_df, pipeline):
    """
    Preprocesses a new data entry using the provided preprocessing pipeline.
    
    Parameters:
    - new_entry_df (pd.DataFrame): New data to preprocess.
    - pipeline (sklearn.pipeline.Pipeline): The preprocessing pipeline.
    
    Returns:
    - X_new_encoded_df (pd.DataFrame): The preprocessed feature set ready for prediction.
    """
    # Assume preprocess_data function is imported or defined elsewhere
    # Here, we'll reuse the preprocess_data function from data_preprocessing.py
    from src.data_preprocessing import preprocess_data
    
    X_new, _, _ = preprocess_data(new_entry_df, pipeline=pipeline, fit_pipeline=False)
    return X_new

def predict_units(model, X_preprocessed):
    """
    Predicts the 'Units' for the preprocessed data using the trained model.
    
    Parameters:
    - model (RandomForestRegressor): The trained ML model.
    - X_preprocessed (pd.DataFrame): The preprocessed feature set.
    
    Returns:
    - predictions (np.ndarray): Predicted 'Units' values.
    """
    predictions = model.predict(X_preprocessed)
    return predictions