"""
MoodLens Flask API
Wraps your existing emotion_model.pkl so server.js can call it.
Run: python flask_api.py
Runs on: http://localhost:5001
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json

app = Flask(__name__)
CORS(app)

# â”€â”€ Paths (same as your app.py) â”€â”€
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_metadata.json")

# â”€â”€ Load model â”€â”€
model    = None
metadata = {}

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("âœ… ML model loaded:", MODEL_PATH)
    except Exception as e:
        print("âŒ Failed to load model:", e)
else:
    print("âš ï¸  No model found at", MODEL_PATH, "â€” will use rule-based fallback")

if os.path.exists(META_PATH):
    try:
        with open(META_PATH) as f:
            metadata = json.load(f)
    except Exception:
        pass

# â”€â”€ Rule-based fallback (copied from your app.py logic) â”€â”€
KEYWORD_RULES = {
    "Happy":      ["love", "great", "excellent", "amazing", "wonderful", "fantastic",
                   "happy", "satisfied", "perfect", "awesome", "thank", "best"],
    "Angry":      ["unacceptable", "furious", "outraged", "terrible", "horrible",
                   "worst", "angry", "disgusting", "incompetent", "sue", "fraud"],
    "Frustrated": ["frustrated", "annoying", "again", "still", "waiting", "useless",
                   "waste", "broken", "never", "ridiculous", "keeps", "repeated"],
    "Sad":        ["sad", "disappointed", "unfortunately", "regret", "miss",
                   "discontinued", "sorry", "unhappy", "lost", "upset"],
    "Excited":    ["excited", "incredible", "can't wait", "amazing", "thrilled",
                   "stoked", "pumped", "early access", "wow", "omg", "so good"],
    "Neutral":    ["received", "arrived", "package", "order", "expected",
                   "noted", "confirmed", "processed", "standard", "okay"],
}

def rule_based_predict(text):
    text_lower = text.lower()
    scores = {emotion: 0 for emotion in KEYWORD_RULES}
    for emotion, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = round(scores[best] / total, 2) if total > 0 else 0.5
    if scores[best] == 0:
        best = "Neutral"
        confidence = 0.5
    return best, confidence

def extract_keywords(text, emotion):
    """Return matching keywords found in text for the predicted emotion."""
    text_lower = text.lower()
    found = []
    if emotion in KEYWORD_RULES:
        for kw in KEYWORD_RULES[emotion]:
            if kw in text_lower:
                found.append(kw)
    return found[:5]  # max 5 keywords

# â”€â”€ Routes â”€â”€

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": metadata.get("best_model", "Naive Bayes") if model else "rule-based"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or not data.get('text', '').strip():
        return jsonify({"error": "Text is required"}), 400

    text = data['text'].strip()

    if model:
        try:
            proba  = model.predict_proba([text])[0]
            classes = model.classes_
            idx    = proba.argmax()
            emotion    = classes[idx]
            confidence = round(float(proba[idx]), 2)
        except Exception as e:
            print("Model prediction failed, using fallback:", e)
            emotion, confidence = rule_based_predict(text)
    else:
        emotion, confidence = rule_based_predict(text)

    keywords = extract_keywords(text, emotion)

    return jsonify({
        "emotion":    emotion,
        "confidence": confidence,
        "keywords":   keywords,
        "model_used": "ml_model" if model else "rule_based"
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
