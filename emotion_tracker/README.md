# 🧠 Customer Emotion Tracker — ML-Powered Edition
**U24IT401 – Artificial Intelligence & Machine Learning | Project Review 3**
Meenakshi Sundararajan Engineering College, Chennai

---

## 👥 Team Members
| Name | Register No |
|------|-------------|
| Amirtha Shree L | 243115205007 |
| Harshawardhini M G | 243115205031 |
| Mahalakshmi P | 243115205057 |

---

## 📁 Project Structure
```
emotion_tracker/
├── app.py                  ← Main Tkinter GUI (run this)
├── train_model.py          ← ML training pipeline (8 datasets)
├── emotion_rules.py        ← Rule-based fallback engine + shared constants
├── datasets/               ← 8 auto-generated CSV training datasets
│   ├── DS1_Ecommerce_Reviews.csv
│   ├── DS2_Social_Media_Complaints.csv
│   ├── DS3_App_Store_Reviews.csv
│   ├── DS4_Hospitality_Reviews.csv
│   ├── DS5_Support_Chat.csv
│   ├── DS6_Healthcare_Feedback.csv
│   ├── DS7_Banking_Finance.csv
│   └── DS8_EdTech_Reviews.csv
└── models/
    ├── emotion_model.pkl       ← Trained ML model
    └── model_metadata.json     ← Accuracy, F1, confusion matrix, dataset stats
```

---

## 🗄️ 8 Training Datasets

| # | Dataset | Domain | Samples |
|---|---------|--------|---------|
| DS1 | Ecommerce Reviews | E-Commerce | 240 |
| DS2 | Social Media Complaints | Social Media / Twitter | 180 |
| DS3 | App Store Reviews | Mobile Apps (iOS/Android) | 210 |
| DS4 | Hospitality Reviews | Hotels & Restaurants | 210 |
| DS5 | Support Chat Transcripts | Customer Support | 240 |
| DS6 | Healthcare Patient Feedback | Healthcare / Clinics | 210 |
| DS7 | Banking & Finance Feedback | Banking & Insurance | 210 |
| DS8 | EdTech Learning Reviews | Education Technology | 210 |
| | **TOTAL** | | **1,710 samples** |

---

## 🤖 ML Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase, strip punctuation/digits/handles)
   ↓
TF-IDF Vectoriser (1–3 grams, 12,000 features, sublinear TF)
   ↓
┌─────────────────────┐
│  3 Models trained:  │
│  • Logistic Regr.  │  F1: 0.9883
│  • Naive Bayes     │  F1: 1.0000  ← Selected (best)
│  • Linear SVM      │  F1: 0.9883
└─────────────────────┘
   ↓
Best model selected by weighted F1
   ↓
Saved → models/emotion_model.pkl
```

---

## 🎯 Emotion Classes
| Emotion | Emoji | Churn Risk |
|---------|-------|------------|
| Happy | 😊 | Low |
| Excited | 🤩 | Low |
| Neutral | 😐 | Medium |
| Sad | 😢 | Medium-High |
| Frustrated | 😤 | High |
| Angry | 😡 | Very High |

---

## 🚀 How to Run

### Step 1 — Install requirements
```bash
pip install scikit-learn numpy pandas joblib
# Optional for polarity boost:
pip install textblob
```

### Step 2 — Train the model (generates datasets + model)
```bash
python train_model.py
```

### Step 3 — Launch the GUI
```bash
python app.py
```

---

## 📊 Model Performance
- **Best Model:** Naive Bayes
- **Accuracy:** 100.00%
- **F1 Score (weighted):** 1.0000
- **Cross-validation:** 99.85% ± 0.29%
- **Train/Test Split:** 80% / 20%

---

## 🖥️ App Features
| Tab | Description |
|-----|-------------|
| 🔍 Analyse | Real-time emotion detection with score breakdown |
| 📊 Dashboard | Live analytics — emotion distribution, churn stats |
| 🧪 Batch Test | Run all 15 samples through the ML engine at once |
| 🤖 ML Metrics | Model accuracy, confusion matrix, dataset summary |
| ℹ️ About | Team info, architecture, SDG goal |

---

*SDG 9 – Industry, Innovation and Infrastructure*
