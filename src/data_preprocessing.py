import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for frequency encoding of high-cardinality categorical features.
    """
    def __init__(self, columns):
        self.columns = columns
        self.frequency_maps = {}
        self.default_values = {}
    
    def fit(self, X, y=None):
        for col in self.columns:
            freq = X[col].value_counts()
            self.frequency_maps[col] = freq.to_dict()
            self.default_values[col] = freq.mean()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            freq_map = self.frequency_maps[col]
            default = self.default_values[col]
            X[col + '_Freq'] = X[col].map(freq_map).fillna(default)
            X.drop(columns=[col], inplace=True)
        return X
    
    def get_feature_names_out(self, input_features=None):
        """
        Returns the feature names after frequency encoding.
        """
        return [f"{col}_Freq" for col in self.columns]

def create_preprocessing_pipeline(high_cardinality_cols=None, low_cardinality_cols=None):
    """
    Creates a preprocessing pipeline with optional frequency encoding, 
    one-hot encoding, and scaling.
    
    Parameters:
    - high_cardinality_cols (list or None): List of high cardinality categorical column names. If None or empty, no frequency encoding is applied.
    - low_cardinality_cols (list or None): List of low cardinality categorical column names. If None or empty, no one-hot encoding is applied.
    
    Returns:
    - sklearn.pipeline.Pipeline: The configured preprocessing pipeline.
    """
    if high_cardinality_cols is None:
        high_cardinality_cols = []
    if low_cardinality_cols is None:
        low_cardinality_cols = []
    
    steps = []
    
    # Add FrequencyEncoder step only if we have high cardinality cols
    if len(high_cardinality_cols) > 0:
        from src.data_preprocessing import FrequencyEncoder  # ensure correct import
        frequency_encoder = FrequencyEncoder(columns=high_cardinality_cols)
        steps.append(('frequency_encoder', frequency_encoder))
    
    # Add OneHotEncoder step only if we have low cardinality cols
    if len(low_cardinality_cols) > 0:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        
        one_hot_encoder = OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False)
        column_transformer = ColumnTransformer(transformers=[
            ('one_hot', one_hot_encoder, low_cardinality_cols)
        ], remainder='passthrough')
        steps.append(('one_hot_encoder', column_transformer))
    
    # Add scaler at the end
    from sklearn.preprocessing import StandardScaler
    steps.append(('scaler', StandardScaler()))
    
    preprocessing_pipeline = Pipeline(steps=steps)
    
    return preprocessing_pipeline

def preprocess_data(df, pipeline=None, fit_pipeline=True):
    """
    Preprocesses the input DataFrame using the provided pipeline.
    
    Parameters:
    - df (pd.DataFrame): The input data to preprocess.
    - pipeline (sklearn.pipeline.Pipeline or None): Existing preprocessing pipeline. If None and fit_pipeline=True, a new pipeline is created and fitted.
    - fit_pipeline (bool): Whether to fit the pipeline on the data.
    
    Returns:
    - X (pd.DataFrame): Preprocessed feature DataFrame.
    - y (pd.Series): Target variable.
    - pipeline (sklearn.pipeline.Pipeline): The fitted preprocessing pipeline.
    """
    # Define columns to drop based on initial sanitization
    columns_to_drop = [
        'Row ID', 'Order ID', 'Customer ID', 'Product ID', 
        'Product Name', 'Division', 'Postal Code'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle Date Columns
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['Order Date'])
    
    # Extract Date Features
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Day'] = df['Order Date'].dt.day
    df['DayOfWeek'] = df['Order Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['WeekOfYear'] = df['Order Date'].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 1 if Saturday or Sunday
    
    # Drop original 'Order Date' and 'Ship Date'
    df = df.drop(columns=['Order Date', 'Ship Date'], errors='ignore')

    y = df['Units'] if 'Units' in df.columns else None
    df = df.drop(columns=['Units'], errors='ignore')  # Drop Units from the features

    # Define categorical columns
    high_cardinality_cols = ['City', 'State/Province']
    low_cardinality_cols = ['Ship Mode', 'Country/Region', 'Region']
    
    # Initialize pipeline if not provided
    if pipeline is None:
        pipeline = create_preprocessing_pipeline(high_cardinality_cols, low_cardinality_cols)
    
    # Fit and transform or just transform
    if fit_pipeline:
        X = pipeline.fit_transform(df)
    else:
        X = pipeline.transform(df)
    
    one_hot_features = pipeline.named_steps['one_hot_encoder'].named_transformers_['one_hot'].get_feature_names_out(low_cardinality_cols)
    freq_features = [f"{col}_Freq" for col in high_cardinality_cols]
    numeric_features = ['Sales', 'Gross Profit', 'Cost', 'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend']
    final_feature_names = list(one_hot_features) + freq_features + numeric_features

    # Create the DataFrame with manually defined columns
    X_encoded_df = pd.DataFrame(X, index=df.index, columns=final_feature_names)
    
    return X_encoded_df, y, pipeline

def preprocess_data_monthly(df, pipeline=None, fit_pipeline=True):
    """
    Preprocesses the monthly-aggregated DataFrame.
    Expects: ['Order Date', 'Units', 'Year', 'Month']

    Parameters:
    - df (pd.DataFrame): Monthly data.
    - pipeline (Pipeline or None): Existing pipeline. If None and fit_pipeline=True, create new.
    - fit_pipeline (bool): Fit the pipeline on the given data if True.

    Returns:
    - X (pd.DataFrame): Preprocessed features.
    - y (pd.Series): Target variable ('Units').
    - pipeline (Pipeline): The fitted pipeline.
    """
    y = df['Units'] if 'Units' in df.columns else None
    df = df.drop(columns=['Units'], errors='ignore')  # Drop Units from the features

    # Numeric features only (already aggregated monthly)
    numeric_features = ['Year', 'Month']

    if pipeline is None:
        pipeline = create_preprocessing_pipeline(high_cardinality_cols=[], low_cardinality_cols=[])

    if fit_pipeline:
        X = pipeline.fit_transform(df[numeric_features])
    else:
        X = pipeline.transform(df[numeric_features])

    X_encoded_df = pd.DataFrame(X, index=df.index, columns=numeric_features)
    return X_encoded_df, y, pipeline
