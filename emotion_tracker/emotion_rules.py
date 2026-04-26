"""
emotion_rules.py  —  Rule-based emotion engine (fallback + constants)
Shared by train_model.py and app.py
"""

import re, random

try:
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

EMOTION_KEYWORDS = {
    "Happy":      ["great","excellent","awesome","love","wonderful","fantastic","happy","good",
                   "pleased","delighted","satisfied","amazing","best","perfect","brilliant",
                   "enjoyed","smooth","easy","fast","quick","helpful","recommend","thank",
                   "impressed","superb","outstanding","brilliant","lovely","smooth","pleased"],
    "Angry":      ["terrible","awful","horrible","hate","disgusting","furious","angry","outraged",
                   "worst","useless","garbage","trash","ridiculous","unacceptable","pathetic",
                   "scam","fraud","never again","waste","rude","incompetent","disgusted",
                   "furious","theft","criminal","disgraceful","appalling","shameful"],
    "Frustrated": ["frustrating","annoyed","disappointed","slow","broken","crashed","failed",
                   "error","bug","issue","problem","keeps","again","still","waiting",
                   "took forever","hours","days","not working","couldn't","unable","keeps crashing",
                   "ridiculous","circles","same","repeated","third","fourth","keeps"],
    "Sad":        ["sad","unfortunate","sorry","regret","miss","lost","disappointed","bad experience",
                   "let down","upset","heartbroken","unhappy","poor","wish","hoped","expected more",
                   "devastated","regret","loss","down","low","unfortunate","shame","miss"],
    "Excited":    ["excited","thrilled","cant wait","amazing","wow","incredible","mind blowing",
                   "game changer","revolutionary","love it","obsessed","must have","blown away",
                   "exceeded","surprised","stunned","outstanding","phenomenal","blown",
                   "changed","incredible","breathtaking","wow","omg","cannot believe"],
    "Neutral":    ["okay","fine","average","normal","standard","regular","usual","moderate",
                   "acceptable","adequate","decent","fair","reasonable","ordinary","typical",
                   "processed","received","completed","acknowledged","noted"],
}

EMOJI_MAP = {
    "Happy":      "😊",
    "Angry":      "😡",
    "Frustrated": "😤",
    "Sad":        "😢",
    "Excited":    "🤩",
    "Neutral":    "😐",
}

COLOR_MAP = {
    "Happy":      "#2ecc71",
    "Angry":      "#e74c3c",
    "Frustrated": "#e67e22",
    "Sad":        "#3498db",
    "Excited":    "#9b59b6",
    "Neutral":    "#95a5a6",
}

CHURN_RISK = {
    "Happy":      "Low",
    "Excited":    "Low",
    "Neutral":    "Medium",
    "Sad":        "Medium-High",
    "Frustrated": "High",
    "Angry":      "Very High",
}

CHURN_COLOR = {
    "Low":         "#2ecc71",
    "Medium":      "#f39c12",
    "Medium-High": "#e67e22",
    "High":        "#e74c3c",
    "Very High":   "#c0392b",
}

SUGGESTIONS = {
    "Happy":      "✅ Customer is satisfied. Great time to request a review or upsell.",
    "Excited":    "🚀 Customer is enthusiastic! Share new features or loyalty rewards.",
    "Neutral":    "📋 Customer is indifferent. Engage with personalised offers.",
    "Sad":        "💬 Reach out with empathy. Understand what went wrong.",
    "Frustrated": "⚡ Escalate to support immediately. Offer resolution & follow-up.",
    "Angry":      "🚨 URGENT: Contact customer within 1 hour. Offer compensation.",
}


def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def predict_emotion(text: str) -> dict:
    words = preprocess(text)
    word_set = set(words)
    bigrams  = {" ".join(words[i:i+2]) for i in range(len(words)-1)}
    trigrams = {" ".join(words[i:i+3]) for i in range(len(words)-2)}
    all_ngrams = word_set | bigrams | trigrams

    scores = {e: 0 for e in EMOTION_KEYWORDS}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in all_ngrams:
                scores[emotion] += 1

    polarity = 0.0
    if NLP_AVAILABLE:
        try:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.3:
                scores["Happy"]   += 2
                scores["Excited"] += 1
            elif polarity < -0.5:
                scores["Angry"]      += 2
                scores["Frustrated"] += 1
            elif polarity < -0.2:
                scores["Sad"]        += 1
                scores["Frustrated"] += 1
        except Exception:
            pass

    total = sum(scores.values()) or 1
    best  = max(scores, key=scores.get)

    if scores[best] == 0:
        best = "Happy" if polarity > 0.1 else "Frustrated" if polarity < -0.1 else "Neutral"

    raw_conf   = scores[best] / total if total else 0
    confidence = min(int(50 + raw_conf * 50), 99)
    if scores[best] == 0:
        confidence = random.randint(52, 67)

    return {
        "emotion":    best,
        "confidence": confidence,
        "scores":     {e: float(s) for e, s in scores.items()},
        "polarity":   round(polarity, 3),
        "emoji":      EMOJI_MAP[best],
        "color":      COLOR_MAP[best],
        "churn_risk": CHURN_RISK[best],
        "suggestion": SUGGESTIONS[best],
        "model_used": "Rule-based Engine",
    }
