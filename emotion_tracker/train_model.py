"""
╔══════════════════════════════════════════════════════════════╗
║   CUSTOMER EMOTION TRACKER — ML TRAINING PIPELINE           ║
║   8 Synthetic Datasets  +  TF-IDF + Logistic Regression     ║
║   U24IT401 – AI & ML  |  Meenakshi Sundararajan Engg College ║
╚══════════════════════════════════════════════════════════════╝
"""

import json, pickle, os, random, re
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import joblib

EMOTIONS = ["Happy", "Angry", "Frustrated", "Sad", "Excited", "Neutral"]
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  8 DATASETS  (each ~200–400 samples, varied domains & styles)
# ═══════════════════════════════════════════════════════════════

def build_all_datasets():
    datasets = {}

    # ── DATASET 1 : E-commerce Product Reviews ────────────────
    ds1 = {
        "name": "DS1_Ecommerce_Reviews",
        "domain": "E-Commerce",
        "description": "Online shopping product reviews with star ratings",
        "samples": []
    }
    templates = {
        "Happy": [
            "Absolutely love this product! Great quality and fast shipping.",
            "Exceeded my expectations. Will definitely buy again!",
            "Perfect product, exactly as described. Very satisfied.",
            "Amazing value for money. Highly recommend to everyone.",
            "Best purchase I've made this year. Works perfectly.",
            "Great product, great price. Delivered quickly too.",
            "Wonderful quality. Packaging was excellent and item arrived safe.",
            "Very pleased with this purchase. Looks exactly like the photos.",
            "Good product and arrived on time. Would buy again.",
            "Excellent! Just what I needed. Five stars from me.",
        ],
        "Angry": [
            "Absolute garbage. Broke within two days. Scam product!",
            "Worst purchase ever. Do not waste your money on this.",
            "This is fraud! Product is completely different from description.",
            "Terrible quality. Returned immediately. Seller is a scammer.",
            "Never buying from this seller again. Complete waste of money.",
            "Horrible product. Smells bad and looks nothing like photos.",
            "Disgusting quality. Seller should be banned from this platform.",
            "Outrageous! They sent me a broken item and refuse refund.",
            "Furious! Three weeks late and product arrived damaged.",
            "Absolutely unacceptable. Reported to consumer forum.",
        ],
        "Frustrated": [
            "Package still hasn't arrived after three weeks. Ridiculous.",
            "This product keeps breaking after one use. So annoying.",
            "Third time ordering and third time getting wrong item.",
            "Customer support is useless. Still no solution after 5 emails.",
            "Cheap material. Stitching fell apart on first wash.",
            "Size was completely wrong despite following the size chart.",
            "App won't connect to this device no matter what I try.",
            "Battery drains within an hour. Not what was advertised.",
            "Keeps stopping after 10 minutes. Completely frustrating.",
            "Waited all day for delivery that never came.",
        ],
        "Sad": [
            "Really disappointed. I had high hopes for this product.",
            "Sad to say this doesn't work as advertised at all.",
            "Unfortunate experience. Quality has really gone downhill.",
            "I miss the old version. This new one is so much worse.",
            "Heartbroken. Bought this as a gift and it arrived broken.",
            "Wish I had read reviews first. Very let down by this.",
            "Such a shame. The concept is great but execution is poor.",
            "Regret buying this. Could have spent money on something better.",
            "Disappointed with the build quality compared to older models.",
            "Expected more from a brand I used to trust.",
        ],
        "Excited": [
            "Oh my god this is AMAZING! Best thing I've ever bought!",
            "I'm literally obsessed with this product! Can't stop using it!",
            "WOW! Blown away by the quality. Exceeded all expectations!",
            "This is a total game changer! Why didn't I buy this sooner?",
            "Mind blowing product! Sharing with everyone I know!",
            "Incredible! This product changed my daily routine completely.",
            "So thrilled with this purchase! Worth every single penny!",
            "Can't believe how good this is! Absolutely stunned!",
            "Revolutionary! This is the future right here in my hands.",
            "Outstanding! I've already ordered two more as gifts!",
        ],
        "Neutral": [
            "Item arrived as described. Nothing special but does the job.",
            "Okay product. Not amazing but not terrible either.",
            "Average quality. Does what it says on the tin.",
            "Decent product for the price. Would consider buying again.",
            "Standard quality. Met my basic requirements.",
            "Fine product. Nothing to rave about but works adequately.",
            "Acceptable quality. Delivery was on time.",
            "Regular product. Meets expectations, nothing more.",
            "It works. That's about all I can say.",
            "Reasonable product at a fair price.",
        ],
    }
    for emotion, texts in templates.items():
        for i, text in enumerate(texts * 4):   # repeat 4x = ~40/emotion
            ds1["samples"].append({"text": text + f" [{i}]", "emotion": emotion})
    datasets["DS1"] = ds1

    # ── DATASET 2 : Social Media Customer Complaints ──────────
    ds2 = {
        "name": "DS2_Social_Media_Complaints",
        "domain": "Social Media / Twitter-style",
        "description": "Short social media posts and brand mentions",
        "samples": []
    }
    sm_templates = {
        "Angry":      [
            "@brand your service is absolutely TERRIBLE. Worst ever! #scam",
            "Disgusted by how @brand handled my complaint. Boycotting them!",
            "Never again @brand! You lost a loyal customer today. Shameful.",
            "@brand sent wrong order AGAIN. Furious doesn't cover it.",
            "Absolutely outraged. @brand charged me twice. Fraud!",
        ],
        "Frustrated": [
            "@brand still no update on my order?? Been 2 weeks!!",
            "3rd time contacting @brand support and still no fix. Ridiculous.",
            "Why is the @brand app crashing EVERY time I open it? Ugh.",
            "@brand internet keeps dropping. This is unacceptable!",
            "Called @brand 5 times today. On hold for hours. So done.",
        ],
        "Happy":      [
            "Shoutout to @brand for the amazing customer service today! 💯",
            "Just got my @brand order and WOW the quality is incredible!",
            "@brand delivered early and product is perfect. Thank you!",
            "Had an issue with @brand and they sorted it within the hour. Legend.",
            "So impressed with @brand right now. Top tier support!",
        ],
        "Excited":    [
            "Just got my @brand package and I'm OBSESSED!! 🤩🔥",
            "OMG @brand just dropped new features and I'm living for it!!",
            "CAN'T STOP USING MY NEW @brand PRODUCT. Game changer!!",
            "Told everyone about @brand's new collection. Must buy!!",
            "@brand just made my day! This is absolutely incredible!!",
        ],
        "Sad":        [
            "Really sad that @brand quality has dropped so much lately.",
            "Used to love @brand but not anymore. Very let down.",
            "@brand lost my order and offered no real apology. Upset.",
            "Miss the old @brand. These days it's just not the same.",
            "Feeling quite down after @brand cancelled my subscription.",
        ],
        "Neutral":    [
            "@brand delivery was on time. Package in good condition.",
            "Received my @brand order today. All looks fine.",
            "@brand support responded. Issue is being looked into.",
            "Using @brand service for a month now. It's okay so far.",
            "@brand product works as expected. No complaints.",
        ],
    }
    for emotion, texts in sm_templates.items():
        for i, t in enumerate(texts * 6):
            ds2["samples"].append({"text": t.replace("[", "").replace("]", "") + f" #{i}", "emotion": emotion})
    datasets["DS2"] = ds2

    # ── DATASET 3 : App Store Reviews ─────────────────────────
    ds3 = {
        "name": "DS3_App_Store_Reviews",
        "domain": "Mobile App Reviews",
        "description": "iOS/Android app store user reviews",
        "samples": []
    }
    app_reviews = {
        "Happy": [
            "Love this app! So easy to use and beautifully designed.",
            "Best app I've downloaded this year. Works flawlessly.",
            "Smooth, fast and intuitive. Exactly what I was looking for.",
            "Really impressed. The new update fixed everything.",
            "5 stars. Does exactly what it promises and more.",
        ],
        "Angry": [
            "This app is absolute garbage. Crashes every time I open it.",
            "Deleted after 5 minutes. Useless and full of bugs.",
            "Terrible app. Steals data and bombards you with ads. Uninstalled.",
            "Worst app ever made. Dev should be ashamed.",
            "Never works. Pure waste of time and storage.",
        ],
        "Frustrated": [
            "The login keeps failing even with correct credentials. So annoying.",
            "App randomly signs me out every day. Please fix this bug.",
            "Notifications stopped working after the latest update.",
            "Can't get past the loading screen. Reinstalled three times.",
            "Constantly buffering. Makes the app completely unusable.",
        ],
        "Sad": [
            "Used to be my favourite app. Heartbroken by these changes.",
            "New update ruined everything I loved about this app.",
            "Sad to see the quality drop after the recent redesign.",
            "Disappointed. Premium features removed without notice.",
            "Wish it was still like the old version. Miss it.",
        ],
        "Excited": [
            "OH WOW this app is AMAZING!! Completely blown away!",
            "Downloaded yesterday and already addicted. Life changing!",
            "This app is revolutionary! Telling everyone about it!",
            "Can't believe something this good is free! Incredible!",
            "The new AI features are mind blowing. Best update ever!",
        ],
        "Neutral": [
            "App works fine. Nothing special but gets the job done.",
            "Decent app. Does what it says. Average experience.",
            "OK app. Interface is a bit clunky but functional.",
            "Works as expected. Not excited but not disappointed.",
            "Standard app. Meets basic needs.",
        ],
    }
    for emotion, texts in app_reviews.items():
        for i, t in enumerate(texts * 7):
            ds3["samples"].append({"text": t + f" v{i}", "emotion": emotion})
    datasets["DS3"] = ds3

    # ── DATASET 4 : Hotel & Restaurant Reviews ────────────────
    ds4 = {
        "name": "DS4_Hospitality_Reviews",
        "domain": "Hotels & Restaurants",
        "description": "TripAdvisor-style hospitality experience reviews",
        "samples": []
    }
    hospitality = {
        "Happy": [
            "Fantastic hotel! Staff were incredibly welcoming and room was spotless.",
            "Best meal I've had in years. Chef deserves a Michelin star!",
            "Wonderful stay. Every detail was perfect from check-in to checkout.",
            "Lovely restaurant. Food was delicious and service impeccable.",
            "Amazing experience! Will definitely be returning next year.",
        ],
        "Angry": [
            "Dirty room with cockroaches. Absolutely disgusting. Health hazard!",
            "Rude staff who made us feel unwelcome. Never returning.",
            "Food was raw and made us sick. Criminal negligence!",
            "Paid for sea view but faced a brick wall. Pure scam.",
            "Worst hotel I've ever stayed in. Zero stars if I could.",
        ],
        "Frustrated": [
            "AC was broken for our entire 3-night stay. Unbearable heat.",
            "Waited 90 minutes for food that arrived cold and wrong.",
            "WiFi didn't work despite being listed as a key amenity.",
            "Check-in took over an hour. Completely disorganised.",
            "Room wasn't ready at 4pm despite paying for early check-in.",
        ],
        "Sad": [
            "This place used to be wonderful. Standards have really fallen.",
            "Sad to see such a beautiful hotel neglected and run down.",
            "Left feeling quite low. Not the experience we hoped for.",
            "Disappointed for our anniversary trip. Ruined the occasion.",
            "Regret choosing this place for a special celebration.",
        ],
        "Excited": [
            "INCREDIBLE! Best hotel I have ever stayed in!! STUNNING!",
            "The food was out of this world! Absolutely phenomenal!",
            "I'm still buzzing from this experience! Just wow!!",
            "Beyond amazing! Upgraded to suite for free. Heaven!",
            "This place blew my mind completely. Will be back every year!",
        ],
        "Neutral": [
            "Standard hotel. Room was clean and bed was comfortable.",
            "Food was okay. Nothing memorable but nothing terrible.",
            "Average stay. Met our basic needs at a reasonable price.",
            "Decent place to stay for a night. Functional and clean.",
            "Normal hotel experience. No complaints, no highlights.",
        ],
    }
    for emotion, texts in hospitality.items():
        for i, t in enumerate(texts * 7):
            ds4["samples"].append({"text": t + f" #{i}", "emotion": emotion})
    datasets["DS4"] = ds4

    # ── DATASET 5 : Customer Support Chat Transcripts ─────────
    ds5 = {
        "name": "DS5_Support_Chat",
        "domain": "Customer Support Chats",
        "description": "Live chat messages from customer support sessions",
        "samples": []
    }
    chats = {
        "Angry": [
            "I have been waiting THREE hours for a response. This is outrageous!",
            "Your agent was incredibly rude and hung up on me. Disgusting service.",
            "I want a full refund NOW. This is fraud and I'll report you.",
            "Completely unacceptable. I've been a customer for 10 years and this is how you treat me?",
            "Your company is an absolute joke. Never seen such incompetence.",
        ],
        "Frustrated": [
            "I've explained this issue four times now. Can someone actually help?",
            "The same problem keeps happening again and again. Nothing is being fixed.",
            "Still waiting for a callback that was promised two days ago.",
            "Your website won't let me log in no matter what I try.",
            "I'm going in circles. Nobody seems to know how to solve this.",
        ],
        "Happy": [
            "That's brilliant, thank you so much! You've been incredibly helpful.",
            "Wow that was fast! Issue resolved perfectly. You're amazing.",
            "So pleased with how quickly this was sorted. Great service!",
            "Really appreciate the patience and help. Will recommend you.",
            "Problem solved! Couldn't be happier with the support.",
        ],
        "Sad": [
            "I'm really upset about losing all my data with no warning.",
            "This has been a very stressful experience. Feeling quite low.",
            "I trusted your company and now I feel completely let down.",
            "It's really disappointing that this keeps happening to me.",
            "I'm sorry to say I'll have to look for alternatives.",
        ],
        "Excited": [
            "Oh that's fantastic news! I'm so relieved and happy now!",
            "YES! That's exactly what I needed! You've made my day!!",
            "Amazing! I can't believe it was that simple. Thank you!",
            "That's brilliant! Can't wait to try the new features!",
            "Wonderful! This solved everything! I'm telling all my friends!",
        ],
        "Neutral": [
            "Okay, I understand. Please process the request when ready.",
            "That's fine. Let me know once it's been updated.",
            "Acknowledged. I'll wait for the confirmation email.",
            "Understood. I'll try that and see if it works.",
            "All good. Thanks for the information.",
        ],
    }
    for emotion, texts in chats.items():
        for i, t in enumerate(texts * 8):
            ds5["samples"].append({"text": t + f" [chat_{i}]", "emotion": emotion})
    datasets["DS5"] = ds5

    # ── DATASET 6 : Healthcare / Clinic Patient Feedback ──────
    ds6 = {
        "name": "DS6_Healthcare_Feedback",
        "domain": "Healthcare",
        "description": "Patient experience feedback for clinics and hospitals",
        "samples": []
    }
    healthcare = {
        "Happy": [
            "Doctor was kind, thorough and made me feel completely at ease.",
            "Excellent care from all staff. Quick appointment and great advice.",
            "So grateful for the professional and compassionate treatment.",
            "Wonderful experience. Diagnosed quickly and treatment worked perfectly.",
            "Best clinic I've visited. Staff are friendly and highly knowledgeable.",
        ],
        "Angry": [
            "Waited three hours with no update. Complete disregard for patients.",
            "Doctor dismissed my symptoms without even examining me. Disgusting.",
            "Billed twice for one appointment. This is highway robbery.",
            "Receptionist was rude and unhelpful. Absolutely unacceptable.",
            "Misdiagnosed and sent home. Had to rush to emergency later.",
        ],
        "Frustrated": [
            "Can't get through on the phone. Waited 40 minutes on hold.",
            "Appointment was changed three times with no explanation.",
            "Prescription wasn't sent to pharmacy despite being told it was.",
            "Test results still not shared after two weeks of chasing.",
            "Online booking system never works. So frustrating.",
        ],
        "Sad": [
            "Came in feeling unwell and left feeling ignored and dismissed.",
            "Difficult news delivered without any empathy. Felt very alone.",
            "Sad to see how overwhelmed and understaffed the clinic is.",
            "Lost faith in this practice after years of being a patient here.",
            "Unfortunate experience during a very vulnerable time in my life.",
        ],
        "Excited": [
            "Results came back all clear! So relieved and grateful for amazing care!",
            "Incredible doctor! Solved a problem I've had for years in one visit!",
            "New treatment is working wonders! Feeling better than ever!",
            "So happy with the care I've received! Truly life changing!",
            "Amazing team! Went above and beyond in every possible way!",
        ],
        "Neutral": [
            "Standard appointment. Doctor was professional and informative.",
            "Waited the usual amount of time. Appointment was straightforward.",
            "Adequate care provided. All questions were answered.",
            "Normal check-up. Nothing unusual to report.",
            "Routine visit. Everything was handled professionally.",
        ],
    }
    for emotion, texts in healthcare.items():
        for i, t in enumerate(texts * 7):
            ds6["samples"].append({"text": t + f" [p{i}]", "emotion": emotion})
    datasets["DS6"] = ds6

    # ── DATASET 7 : Banking & Finance Feedback ────────────────
    ds7 = {
        "name": "DS7_Banking_Finance",
        "domain": "Banking & Financial Services",
        "description": "Customer feedback for banks, loans and insurance",
        "samples": []
    }
    banking = {
        "Angry": [
            "Unauthorised transaction drained my account and bank doesn't care!",
            "Account frozen with zero notice during an emergency. Disgraceful!",
            "Hidden charges nobody told me about. This is theft!",
            "Insurance claim denied unfairly. Taking this to the ombudsman.",
            "Loan approved then cancelled without any reason given. Furious!",
        ],
        "Frustrated": [
            "Transaction limit preventing me from paying my own bills. Ridiculous.",
            "App keeps logging me out mid-transfer. Happens every single time.",
            "Spent 2 hours on hold just to update my address. Unbelievable.",
            "Cheque still not cleared after 12 days. Standard is 5 days.",
            "Three failed card payments due to a system error on your end.",
        ],
        "Happy": [
            "Mortgage advisor was fantastic. Process was smooth and stress-free.",
            "Fraud alert resolved within minutes. Great security team!",
            "New app is brilliant! Managing finances has never been easier.",
            "Interest rate on savings account is the best I've found anywhere.",
            "Outstanding service. Loan approved in 24 hours. Highly recommend!",
        ],
        "Sad": [
            "Declined for a mortgage despite a perfect credit score. Devastated.",
            "Pension fund underperformed badly. Years of savings feel wasted.",
            "Sad to close my account after 20 years. Just can't afford the fees.",
            "Losing trust in financial institutions after this experience.",
            "Difficult situation made worse by a lack of empathy from staff.",
        ],
        "Excited": [
            "Investment portfolio up 40% this quarter! Brilliant fund management!",
            "Got approved for my first mortgage!! Dreams do come true!!",
            "New cashback rewards are incredible! Getting paid to spend!",
            "Just unlocked the platinum tier! Benefits are amazing!",
            "Financial advisor helped me save £5000 this year! Life changing!",
        ],
        "Neutral": [
            "Statement balance is correct. No issues this month.",
            "Transfer processed successfully within expected timeframe.",
            "Card replacement arrived in 5 working days as stated.",
            "Interest calculated correctly on savings account.",
            "Direct debits all processed on time as scheduled.",
        ],
    }
    for emotion, texts in banking.items():
        for i, t in enumerate(texts * 7):
            ds7["samples"].append({"text": t + f" [acct_{i}]", "emotion": emotion})
    datasets["DS7"] = ds7

    # ── DATASET 8 : EdTech / Online Learning Feedback ─────────
    ds8 = {
        "name": "DS8_EdTech_Reviews",
        "domain": "Education Technology",
        "description": "Student feedback for online courses and learning platforms",
        "samples": []
    }
    edtech = {
        "Happy": [
            "Best course I've ever taken. Instructor explained everything so clearly.",
            "Learned more in this course than in 3 years of college. Amazing!",
            "Content is excellent and well-structured. Highly recommended.",
            "Fantastic platform. So intuitive and packed with great resources.",
            "Passed my exam with distinction thanks to this course!",
        ],
        "Angry": [
            "Paid for a certificate that doesn't work on LinkedIn. Total scam!",
            "Course videos won't load at all. Demanding a full refund.",
            "Instructor is completely absent and never answers questions. Fraud!",
            "Promised live sessions that were pre-recorded. Misleading!",
            "Downloaded materials were corrupted. No support. Disgusting.",
        ],
        "Frustrated": [
            "Video player keeps buffering despite good internet connection.",
            "Quiz doesn't accept correct answers due to a technical bug.",
            "Progress isn't saving so I keep restarting the same lessons.",
            "Certificate still not issued 3 weeks after course completion.",
            "Can't access downloaded content offline despite it being advertised.",
        ],
        "Sad": [
            "Course was cancelled halfway through with no refund offered.",
            "Instructor left the platform and no replacement was provided.",
            "Sad to see content quality drop so much after the rebranding.",
            "Put months of effort into this course only for it to be delisted.",
            "Expected so much more from a platform with such great reviews.",
        ],
        "Excited": [
            "Just completed my first course and got a job offer! AMAZING!!",
            "This platform completely changed my career! Incredible content!",
            "Mind blowing instructors! Learning has never been this fun!",
            "Finished the bootcamp and built my first app! So proud and thrilled!",
            "This course opened so many doors. I'm obsessed with learning now!",
        ],
        "Neutral": [
            "Course covered the basics adequately. Nothing groundbreaking.",
            "Videos are clear and content is organised logically.",
            "Standard online learning experience. Does what it promises.",
            "Completed the module. Assessment was straightforward.",
            "Platform works well. Interface is clean and easy to navigate.",
        ],
    }
    for emotion, texts in edtech.items():
        for i, t in enumerate(texts * 7):
            ds8["samples"].append({"text": t + f" [edu_{i}]", "emotion": emotion})
    datasets["DS8"] = ds8

    return datasets


# ═══════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def preprocess(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]|#\w+|@\w+", " ", text)   # strip tags/handles
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    return " ".join(text.split())

# ═══════════════════════════════════════════════════════════════
#  REAL DATASET LOADER — Emotions NLP (train / val / test)
# ═══════════════════════════════════════════════════════════════

# Map NLP emotion labels → your 6 project emotions
EMOTION_MAP = {
    "joy":      "Happy",
    "anger":    "Angry",
    "sadness":  "Sad",
    "fear":     "Frustrated",
    "surprise": "Excited",
    "love":     "Happy",    # love → Happy (merged)
}

def load_txt_file(filepath):
    """Load a semicolon-separated .txt file → list of (text, emotion) tuples."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            parts = line.rsplit(";", 1)        # split from right
            if len(parts) != 2:
                continue
            text, label = parts[0].strip(), parts[1].strip().lower()
            emotion = EMOTION_MAP.get(label)
            if emotion:                         # skip unmapped labels
                samples.append({"text": text, "emotion": emotion})
    return samples


def load_real_datasets():
    """Load train.txt, val.txt, test.txt from the datasets/ folder."""
    print("\n" + "═"*60)
    print("  LOADING REAL DATASETS (Emotions NLP)")
    print("═"*60)

    files = {
        "train": os.path.join(DATASET_DIR, "train.txt"),
        "val":   os.path.join(DATASET_DIR, "val.txt"),
        "test":  os.path.join(DATASET_DIR, "test.txt"),
    }

    # Check all files exist
    for split, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"  ❌ Missing: {path}\n"
                f"  Please place train.txt, val.txt, test.txt inside your datasets/ folder."
            )

    all_samples = []
    for split, path in files.items():
        samples = load_txt_file(path)
        print(f"  {split}.txt → {len(samples)} samples loaded")
        all_samples.extend(samples)

    # Show emotion distribution
    from collections import Counter
    counts = Counter(s["emotion"] for s in all_samples)
    print(f"\n  Total samples : {len(all_samples)}")
    print(f"  Emotion distribution:")
    for emotion, count in sorted(counts.items()):
        bar = "█" * (count // 200)
        print(f"    {emotion:<12} {count:>5}  {bar}")

    # Save as CSV for EDA
    df = pd.DataFrame(all_samples)
    df["text_clean"] = df["text"].apply(preprocess)
    csv_path = os.path.join(DATASET_DIR, "real_emotions_combined.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Saved combined CSV → {csv_path}")
    print("═"*60)

    return all_samples


def train_model(datasets, model_type="lr"):
    print("\n" + "═"*60)
    print("  BUILDING COMBINED TRAINING CORPUS")
    print("═"*60)

    all_texts, all_labels = [], []
    dataset_stats = {}

    for key, ds in datasets.items():
        samples = ds["samples"]
        counts = Counter(s["emotion"] for s in samples)
        dataset_stats[key] = {
            "name":   ds["name"],
            "domain": ds["domain"],
            "total":  len(samples),
            "counts": dict(counts),
        }
        print(f"  {key}: {ds['name']}")
        print(f"       Domain : {ds['domain']}")
        print(f"       Samples: {len(samples)}  |  {dict(counts)}")
        for s in samples:
            all_texts.append(preprocess(s["text"]))
            all_labels.append(s["emotion"])

    print(f"\n  ✅ Total combined samples : {len(all_texts)}")
    print(f"  ✅ Emotion classes        : {sorted(set(all_labels))}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    print(f"  ✅ Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Model selection ──
    print("\n" + "═"*60)
    print("  TRAINING MODELS")
    print("═"*60)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=12000,
        sublinear_tf=True,
        min_df=2,
    )

    classifiers = {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf),
            ("clf",   LogisticRegression(max_iter=1000, C=5.0, random_state=42,
                                          solver="lbfgs")),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=12000, sublinear_tf=True)),
            ("clf",   LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]),
    }

    results = {}
    best_name, best_score, best_model = None, 0, None

    for name, pipeline in classifiers.items():
        print(f"\n  Training: {name} …", end=" ")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        print(f"done.")
        print(f"       Accuracy : {acc*100:.2f}%  |  F1 (weighted): {f1:.4f}")
        print(f"       CV Score : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        results[name] = {"accuracy": acc, "f1": f1, "cv_mean": cv_scores.mean(),
                          "cv_std": cv_scores.std(), "y_pred": y_pred}
        if f1 > best_score:
            best_score = f1
            best_name  = name
            best_model = pipeline

    print(f"\n  🏆 Best Model : {best_name}  (F1={best_score:.4f})")

    # ── Full classification report for best model ──
    print("\n" + "═"*60)
    print(f"  CLASSIFICATION REPORT  [{best_name}]")
    print("═"*60)
    print(classification_report(y_test, results[best_name]["y_pred"],
                                  target_names=sorted(set(all_labels))))

    # ── Save model + metadata ──
    model_path = os.path.join(MODEL_DIR, "emotion_model.pkl")
    meta_path  = os.path.join(MODEL_DIR, "model_metadata.json")

    joblib.dump(best_model, model_path)
    print(f"  ✅ Model saved → {model_path}")

    # confusion matrix as dict
    labels_sorted = sorted(set(all_labels))
    cm = confusion_matrix(y_test, results[best_name]["y_pred"], labels=labels_sorted)

    metadata = {
        "trained_at":    datetime.now().isoformat(),
        "best_model":    best_name,
        "best_f1":       round(best_score, 4),
        "best_accuracy": round(results[best_name]["accuracy"], 4),
        "total_samples": len(all_texts),
        "train_size":    len(X_train),
        "test_size":     len(X_test),
        "emotions":      labels_sorted,
        "datasets":      dataset_stats,
        "all_results":   {k: {"accuracy": round(v["accuracy"],4),
                               "f1":       round(v["f1"],4),
                               "cv_mean":  round(v["cv_mean"],4),
                               "cv_std":   round(v["cv_std"],4)}
                          for k, v in results.items()},
        "confusion_matrix": {
            "labels":  labels_sorted,
            "matrix":  cm.tolist(),
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata saved → {meta_path}")

    # Save individual dataset CSVs
    for key, ds in datasets.items():
        df = pd.DataFrame(ds["samples"])
        df["text_clean"] = df["text"].apply(preprocess)
        csv_path = os.path.join(DATASET_DIR, f"{ds['name']}.csv")
        df.to_csv(csv_path, index=False)
    print(f"  ✅ Dataset CSVs saved → {DATASET_DIR}/")

    print("\n" + "═"*60)
    print("  TRAINING COMPLETE")
    print("═"*60)
    return best_model, metadata


# ═══════════════════════════════════════════════════════════════
#  INFERENCE  (used by the Tkinter app)
# ═══════════════════════════════════════════════════════════════

_model    = None
_metadata = None

def _load_model():
    global _model, _metadata
    if _model is None:
        mp = os.path.join(MODEL_DIR, "emotion_model.pkl")
        mm = os.path.join(MODEL_DIR, "model_metadata.json")
        if os.path.exists(mp):
            _model = joblib.load(mp)
            with open(mm) as f:
                _metadata = json.load(f)
    return _model, _metadata


def predict_with_ml(text: str) -> dict:
    """Predict emotion using trained ML model. Falls back to rule-based."""
    from emotion_rules import predict_emotion as rule_predict   # local fallback

    model, meta = _load_model()
    if model is None:
        return rule_predict(text)

    clean = preprocess(text)
    emotion = model.predict([clean])[0]

    # Get probability estimates if available
    confidence = 80
    try:
        proba = model.predict_proba([clean])[0]
        classes = model.classes_
        idx = list(classes).index(emotion)
        confidence = min(int(proba[idx] * 100), 99)
        scores_dict = {c: round(float(p * 10), 2) for c, p in zip(classes, proba)}
    except Exception:
        # LinearSVC has no predict_proba – use decision function
        try:
            df_vals = model.decision_function([clean])[0]
            classes = model.classes_
            scores_dict = {c: round(float(v), 3) for c, v in zip(classes, df_vals)}
            # normalise to 0-10 range
            mn, mx = min(scores_dict.values()), max(scores_dict.values())
            if mx > mn:
                scores_dict = {c: round((v-mn)/(mx-mn)*10, 2) for c,v in scores_dict.items()}
            idx = list(classes).index(emotion)
            confidence = min(int(60 + scores_dict.get(emotion, 5) * 3.5), 99)
        except Exception:
            scores_dict = {e: 0 for e in EMOTIONS}
            scores_dict[emotion] = 5

    from emotion_rules import (EMOJI_MAP, COLOR_MAP, CHURN_RISK,
                                CHURN_COLOR, SUGGESTIONS)
    return {
        "emotion":    emotion,
        "confidence": confidence,
        "scores":     scores_dict,
        "polarity":   0.0,
        "emoji":      EMOJI_MAP[emotion],
        "color":      COLOR_MAP[emotion],
        "churn_risk": CHURN_RISK[emotion],
        "suggestion": SUGGESTIONS[emotion],
        "model_used": meta["best_model"] if meta else "ML Model",
    }


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Toggle this to switch between real and synthetic data ──
    USE_REAL_DATA = True   # ← Set False to use synthetic datasets

    if USE_REAL_DATA:
        samples = load_real_datasets()
        datasets = {
            "REAL": {
                "name":        "Real_Emotions_NLP",
                "domain":      "Multi-domain (NLP Emotions)",
                "description": "Real labeled emotion dataset from Kaggle",
                "samples":     samples,
            }
        }
    else:
        datasets = build_all_datasets()

    model, metadata = train_model(datasets)

    print(f"\n  Dataset breakdown:")
    for k, v in metadata["datasets"].items():
        print(f"   {k}: {v['name']} ({v['total']} samples) – {v['domain']}")
    print(f"\n  Model F1 Score : {metadata['best_f1']}")
    print(f"  Model Accuracy : {metadata['best_accuracy']*100:.2f}%")
    print("\n  Run `python app.py` to launch the Tkinter GUI.\n")