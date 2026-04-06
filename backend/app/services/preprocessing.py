import pandas as pd
from datetime import datetime

def preprocess_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw simulated logs and structures timestamps for downstream processing.
    """
    if df.empty:
        return df

    # Convert timestamp to datetime if not already, catching unparseable strings
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isna().any():
        print("Warning: Unparseable timestamps found. Replacing with valid datetimes to protect models.")
        df.loc[df['timestamp'].isna(), 'timestamp'] = pd.Timestamp.now()
        
    
    # Extract time-based features for easier aggregation
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Fill NaN values for optional columns like file_size_mb
    if 'file_size_mb' in df.columns:
        df['file_size_mb'] = df['file_size_mb'].fillna(0.0)
    else:
        df['file_size_mb'] = 0.0

    if 'is_malicious_simulated' not in df.columns:
        df['is_malicious_simulated'] = False
        
    return df