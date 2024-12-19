import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(X, y, model_type='random_forest', model_params=None):
    """
    Trains a Random Forest Regressor on the provided data.
    
    Parameters:
    - X (pd.DataFrame): Preprocessed feature set.
    - y (pd.Series): Target variable.
    - model_type (str): Type of model to train ('random_forest', 'linear', 'xgboost', etc.)
    - model_params (dict or None): Model hyperparameters. If None, defaults are used.
    
    Returns:
    - model (RandomForestRegressor): The trained Random Forest model.
    """
    if model_type == 'random_forest':
        if model_params is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(**model_params)
    elif model_type == 'linear':
        model = LinearRegression() if model_params is None else LinearRegression(**model_params)

    # elif model_type == 'xgboost':
    #     model = XGBRegressor(**model_params) if model_params else XGBRegressor()

    # Add more models as needed
    
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    
    Parameters:
    - model (RandomForestRegressor): The trained model.
    - X_test (pd.DataFrame): Test feature set.
    - y_test (pd.Series): Test target variable.
    
    Returns:
    - metrics (dict): Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Root Mean Squared Error (RMSE)': rmse,
        'R² Score': r2
    }
    
    return metrics

def save_model(model, filepath):
    """
    Saves the trained model to disk.
    
    Parameters:
    - model (RandomForestRegressor): The trained model.
    - filepath (str): Destination file path.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """
    Loads a trained model from disk.
    
    Parameters:
    - filepath (str): Path to the saved model file.
    
    Returns:
    - model (RandomForestRegressor): The loaded model.
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model