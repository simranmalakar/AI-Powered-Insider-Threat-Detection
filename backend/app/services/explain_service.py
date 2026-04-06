import shap
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

def generate_shap_explanation(model, X_train: pd.DataFrame, instance: pd.DataFrame):
    """
    Generates SHAP values for a specific instance using the provided model.
    Since IsolationForest in sklearn doesn't calculate probabilities directly in a way SHAP TreeExplainer prefers without caveats,
    we can use a KernelExplainer or roughly explain the decision function.
    Given the performance constraints of real-time streaming, we'll try TreeExplainer for IF, 
    or a simple permutation explainer if it's black box.
    """
    # Create an explainer
    # For Isolation Forest, SHAP TreeExplainer works directly on the decision_function.
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation='interventional', data=shap.sample(X_train, 50))
        shap_values = explainer.shap_values(instance)
        
        # Calculate feature importance for this instance
        # TreeExplainer on IF returns standard SHAP values
        feature_names = instance.columns.tolist()
        val = shap_values[0] if isinstance(shap_values, list) else shap_values
        if len(val.shape) > 1:
            val = val[0]
            
        contributions = {feature_names[i]: float(val[i]) for i in range(len(feature_names))}
        
        # Sort by absolute impact
        sorted_contributions = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
        return sorted_contributions

    except Exception as e:
        # Fallback to feature weights or dummy if SHAP fails during online prediction
        print(f"SHAP Explainer Error: {str(e)}")
        # Simple dummy explanation based on value magnitudes if SHAP fails
        vals = instance.iloc[0].to_dict()
        return {k: float(v) * 0.1 for k, v in vals.items()}


def generate_lime_explanation(model, X_train: pd.DataFrame, instance: pd.DataFrame):
    """
    Generates LIME feature importances for a specific instance.
    """
    try:
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Score'],
            mode='regression' # Using decision_function which returns continuous scores
        )
        
        # Explain the instance
        exp = explainer.explain_instance(
            data_row=instance.iloc[0].values, 
            predict_fn=model.decision_function, # decision_function returns anomaly scores
            num_features=len(X_train.columns)
        )
        
        # Convert to dictionary {feature: weight}
        contributions = {feat: float(weight) for feat, weight in exp.as_list()}
        return contributions

    except Exception as e:
        print(f"LIME Explainer Error: {str(e)}")
        vals = instance.iloc[0].to_dict()
        return {k: float(v) * 0.1 for k, v in vals.items()}
