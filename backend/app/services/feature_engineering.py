import pandas as pd
import numpy as np
from app.services.llm_service import analyze_text_intent

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms preprocessed logs into a feature matrix suitable for ML models.
    Aggregates data by user over a time window (e.g., daily or per session).
    Since our simulation is small, we'll aggregate by user for the whole simulated period,
    or we can score each event individually. For real-time streaming, event-based profiling is better.
    We will create an event-level feature row.
    """
    if df.empty:
        return df

    features = []
    
    for idx, row in df.iterrows():
        # Baseline features
        record = {
            'log_id': idx,
            'user': row['user'],
            'timestamp': row['timestamp'],
            'is_malicious': row.get('is_malicious_simulated', False),
            'event_type': row['event_type']
        }
        
        # 1. Temporal / Frequency Features
        # Is the hour outside normal business hours (8 AM - 6 PM)?
        is_off_hours = 1 if (row['hour'] < 8 or row['hour'] > 18) else 0
        record['feat_off_hours'] = is_off_hours
        record['feat_is_weekend'] = row['is_weekend']
        
        # 2. Behavioral / Event Specific Features
        record['feat_is_file_access'] = 1 if row['event_type'] == 'file_access' else 0
        record['feat_is_usb'] = 1 if row['event_type'] == 'usb_connect' else 0
        record['feat_is_email'] = 1 if row['event_type'] == 'email' else 0
        
        # 3. Anomaly / Threshold Features
        # E.g., unusual file sizes
        file_size = row.get('file_size_mb', 0)
        record['feat_file_size'] = np.log1p(file_size) # log scaled
        
        # 4. NLP / LLM Features (Mocked via llm_service)
        # We simulate checking email text with an LLM for intent
        intent_score = 0.0
        if row['event_type'] == 'email':
            intent_score = analyze_text_intent(row.get('details', ''))
        record['feat_nlp_intent_score'] = intent_score
        
        features.append(record)
        
    features_df = pd.DataFrame(features)
    return features_df

def get_feature_matrix(features_df: pd.DataFrame):
    """
    Extracts only the numerical feature columns for model training/inference.
    """
    feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
    return features_df[feature_cols].copy()