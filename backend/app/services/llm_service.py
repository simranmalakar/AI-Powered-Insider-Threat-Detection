import os

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import pipeline
    # Use a small zero-shot model for quick inference
    print("Loading LLM model for intent detection...")
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
except ImportError:
    classifier = None
    print("Transformers not installed. Falling back to heuristic.")
except Exception as e:
    classifier = None
    print(f"Error loading model: {e}")

def analyze_text_intent(text: str) -> float:
    """
    Uses a zero-shot classification LLM to estimate "malicious intent".
    Returns a score between 0.0 (safe) and 1.0 (highly suspicious).
    Falls back to heuristics if the model fails to load.
    """
    if not text:
        return 0.0
        
    if classifier is None:
        text_lower = text.lower()
        suspicious_keywords = [
            'resignation', 'confidential', 'source code', 'export', 
            'password', 'wire transfer', 'urgent', 'ip', 'customer database'
        ]
        score = 0.0
        for keyword in suspicious_keywords:
            if keyword in text_lower:
                score += 0.3
        return min(1.0, score)

    try:
        candidate_labels = ["malicious intent", "data theft", "resignation", "normal business"]
        result = classifier(text, candidate_labels)
        
        score = 0.0
        # Sum the probabilities of the suspicious labels
        for label, s in zip(result['labels'], result['scores']):
            if label in ["malicious intent", "data theft", "resignation"]:
                score += s
        return min(1.0, score)
    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return 0.5
