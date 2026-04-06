import pandas as pd
from sklearn.ensemble import IsolationForest

def train_isolation_forest(X: pd.DataFrame) -> IsolationForest:
    """
    Trains an Isolation Forest anomaly detection model.
    """
    # Using a contamination rate roughly equal to the expected anomaly rate
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)
    return model

def predict_isolation_forest(model: IsolationForest, X: pd.DataFrame) -> pd.Series:
    """
    Returns anomaly scores from the IF model.
    In sklearn, negative scores are anomalies. We invert and normalize to 0-1 range.
    """
    # smaller scores are more anomalous. We'll invert them for risk representation
    scores = model.decision_function(X)
    # Scale scores between 0 and 1, where 1 is highly anomalous.
    # Typical IF scores are between -0.5 and 0.5.
    
    # Simple Min-Max scaling of inverted scores
    inverted_scores = -scores 
    min_score, max_score = inverted_scores.min(), inverted_scores.max()
    
    if max_score > min_score:
        normalized_scores = (inverted_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = inverted_scores * 0.0
        
    return normalized_scores