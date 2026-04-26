"""
╔══════════════════════════════════════════════════════════════╗
║       CUSTOMER EMOTION TRACKER — GUI APPLICATION             ║
║       With Modern Dark Theme Login Page                      ║
║       U24IT401 – AI & ML | Meenakshi Sundararajan Engg       ║
╚══════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os
import json

# ── Try importing optional libs ───────────────────────────────
try:
    from emotion_rules import predict_emotion_rule_based, EMOTION_CONFIG
    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_metadata.json")

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE — Dark Theme
# ══════════════════════════════════════════════════════════════
BG_DARK      = "#0F0F1A"   # main background
BG_CARD      = "#1A1A2E"   # card / panel background
BG_INPUT     = "#16213E"   # input field background
BG_HOVER     = "#0F3460"   # button hover
ACCENT       = "#E94560"   # primary accent (red-pink)
ACCENT2      = "#533483"   # secondary accent (purple)
TEXT_PRIMARY = "#EAEAEA"   # main text
TEXT_DIM     = "#8888AA"   # dimmed / placeholder text
TEXT_WHITE   = "#FFFFFF"
BORDER       = "#2A2A4A"   # subtle border
SUCCESS      = "#2ECC71"
WARNING      = "#E67E22"
DANGER       = "#E74C3C"
INFO         = "#3498DB"

# ── Emotion colours ───────────────────────────────────────────
EMOTION_COLORS = {
    "Happy":      "#2ECC71",
    "Angry":      "#E74C3C",
    "Frustrated": "#E67E22",
    "Sad":        "#3498DB",
    "Excited":    "#9B59B6",
    "Neutral":    "#95A5A6",
}

EMOTION_EMOJIS = {
    "Happy":      "😊",
    "Angry":      "😠",
    "Frustrated": "😤",
    "Sad":        "😢",
    "Excited":    "🤩",
    "Neutral":    "😐",
}

CHURN_RISK = {
    "Happy":      ("Low",    SUCCESS),
    "Excited":    ("Low",    SUCCESS),
    "Neutral":    ("Medium", WARNING),
    "Sad":        ("Medium", WARNING),
    "Frustrated": ("High",   DANGER),
    "Angry":      ("High",   DANGER),
}

SUGGESTIONS = {
    "Happy":      "Customer is satisfied. Consider requesting a review or upsell.",
    "Excited":    "High engagement! Great time to introduce loyalty rewards.",
    "Neutral":    "Provide additional information or follow-up to increase satisfaction.",
    "Sad":        "Reach out with empathy. Offer compensation or resolution.",
    "Frustrated": "Escalate to senior support immediately. Prioritise resolution.",
    "Angry":      "Urgent: Contact customer within 1 hour. Offer apology and fix.",
}

# ── Hardcoded credentials (simple, no database) ───────────────
VALID_USERS = {
    "admin":  "admin123",
    "agent":  "agent123",
    "amirtha": "msec2024",
}

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════
model    = None
metadata = {}

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

if os.path.exists(META_PATH):
    try:
        with open(META_PATH) as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}


def predict(text):
    """Predict emotion from text using ML model or rule-based fallback."""
    if not text.strip():
        return None, 0.0

    if model:
        try:
            proba = model.predict_proba([text])[0]
            classes = model.classes_
            idx = proba.argmax()
            return classes[idx], float(proba[idx])
        except Exception:
            pass

    if RULES_AVAILABLE:
        result = predict_emotion_rule_based(text)
        return result.get("emotion", "Neutral"), result.get("confidence", 0.5)

    return "Neutral", 0.5


# ══════════════════════════════════════════════════════════════
#  LOGIN WINDOW
# ══════════════════════════════════════════════════════════════
class LoginWindow:
    def __init__(self, root, on_success):
        self.root = root
        self.on_success = on_success

        self.root.title("Customer Emotion Tracker — Login")
        self.root.geometry("480x580")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_DARK)
        self._center_window(480, 580)

        self._build_ui()

    def _center_window(self, w, h):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth()  // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        # ── Outer frame ──
        outer = tk.Frame(self.root, bg=BG_DARK)
        outer.pack(fill="both", expand=True, padx=40, pady=30)

        # ── Logo / Title area ──
        tk.Label(outer, text="🎯", font=("Arial", 48),
                 bg=BG_DARK, fg=ACCENT).pack(pady=(10, 0))

        tk.Label(outer, text="Customer Emotion Tracker",
                 font=("Arial", 18, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE).pack(pady=(8, 2))

        tk.Label(outer, text="Sign in to continue",
                 font=("Arial", 11),
                 bg=BG_DARK, fg=TEXT_DIM).pack(pady=(0, 24))

        # ── Card ──
        card = tk.Frame(outer, bg=BG_CARD, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        card.pack(fill="x", pady=(0, 16))

        inner = tk.Frame(card, bg=BG_CARD)
        inner.pack(fill="x", padx=28, pady=28)

        # Username
        tk.Label(inner, text="USERNAME", font=("Arial", 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")

        self.username_var = tk.StringVar()
        user_frame = tk.Frame(inner, bg=BG_INPUT, bd=0,
                              highlightthickness=1, highlightbackground=BORDER)
        user_frame.pack(fill="x", pady=(4, 16))

        tk.Label(user_frame, text="  👤  ", font=("Arial", 12),
                 bg=BG_INPUT, fg=TEXT_DIM).pack(side="left")
        self.username_entry = tk.Entry(user_frame, textvariable=self.username_var,
                                       font=("Arial", 12), bg=BG_INPUT,
                                       fg=TEXT_PRIMARY, insertbackground=ACCENT,
                                       relief="flat", bd=0)
        self.username_entry.pack(side="left", fill="x", expand=True,
                                  ipady=10, padx=(0, 10))

        # Password
        tk.Label(inner, text="PASSWORD", font=("Arial", 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")

        self.password_var = tk.StringVar()
        pass_frame = tk.Frame(inner, bg=BG_INPUT, bd=0,
                              highlightthickness=1, highlightbackground=BORDER)
        pass_frame.pack(fill="x", pady=(4, 8))

        tk.Label(pass_frame, text="  🔒  ", font=("Arial", 12),
                 bg=BG_INPUT, fg=TEXT_DIM).pack(side="left")
        self.password_entry = tk.Entry(pass_frame, textvariable=self.password_var,
                                        font=("Arial", 12), bg=BG_INPUT,
                                        fg=TEXT_PRIMARY, insertbackground=ACCENT,
                                        relief="flat", bd=0, show="●")
        self.password_entry.pack(side="left", fill="x", expand=True,
                                  ipady=10, padx=(0, 4))

        # Show/hide password toggle
        self.show_pass = False
        self.eye_btn = tk.Label(pass_frame, text="👁", font=("Arial", 12),
                                 bg=BG_INPUT, fg=TEXT_DIM, cursor="hand2")
        self.eye_btn.pack(side="right", padx=8)
        self.eye_btn.bind("<Button-1>", self._toggle_password)

        # Error label
        self.error_var = tk.StringVar()
        self.error_label = tk.Label(inner, textvariable=self.error_var,
                                     font=("Arial", 10), bg=BG_CARD, fg=ACCENT)
        self.error_label.pack(pady=(4, 0))

        # Login button
        self.login_btn = tk.Button(inner, text="SIGN IN",
                                    font=("Arial", 13, "bold"),
                                    bg=ACCENT, fg=TEXT_WHITE,
                                    activebackground=ACCENT2,
                                    activeforeground=TEXT_WHITE,
                                    relief="flat", bd=0, cursor="hand2",
                                    pady=12, command=self._attempt_login)
        self.login_btn.pack(fill="x", pady=(16, 0))

        # Hover effects on login button
        self.login_btn.bind("<Enter>", lambda e: self.login_btn.config(bg=ACCENT2))
        self.login_btn.bind("<Leave>", lambda e: self.login_btn.config(bg=ACCENT))

        # ── Hint text ──
        hint = tk.Frame(outer, bg=BG_DARK)
        hint.pack(fill="x")
        tk.Label(hint, text="Demo credentials:  admin / admin123   |   agent / agent123",
                 font=("Arial", 9), bg=BG_DARK, fg=TEXT_DIM).pack()

        # ── Course info ──
        tk.Label(outer, text="U24IT401 – AI & ML  |  Meenakshi Sundararajan Engineering College",
                 font=("Arial", 8), bg=BG_DARK, fg=TEXT_DIM).pack(side="bottom", pady=(16, 0))

        # Bind Enter key
        self.root.bind("<Return>", lambda e: self._attempt_login())
        self.username_entry.focus()

    def _toggle_password(self, event=None):
        self.show_pass = not self.show_pass
        self.password_entry.config(show="" if self.show_pass else "●")
        self.eye_btn.config(fg=ACCENT if self.show_pass else TEXT_DIM)

    def _attempt_login(self):
        username = self.username_var.get().strip().lower()
        password = self.password_var.get().strip()

        if not username:
            self._shake_error("Please enter your username.")
            return
        if not password:
            self._shake_error("Please enter your password.")
            return

        if VALID_USERS.get(username) == password:
            self.error_var.set("")
            self.login_btn.config(text="✓  Welcome!", bg=SUCCESS)
            self.root.after(600, lambda: self.on_success(username))
        else:
            self._shake_error("Invalid username or password.")
            self.password_var.set("")
            self.password_entry.focus()

    def _shake_error(self, msg):
        self.error_var.set(f"⚠  {msg}")
        # Simple shake animation
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        for dx in [8, -8, 6, -6, 4, -4, 0]:
            self.root.after(20, lambda d=dx: self.root.geometry(f"+{x+d}+{y}"))


# ══════════════════════════════════════════════════════════════
#  MAIN APP WINDOW
# ══════════════════════════════════════════════════════════════
class EmotionTrackerApp:
    def __init__(self, root, username="user"):
        self.root = root
        self.username = username

        self.root.title("Customer Emotion Tracker")
        self.root.geometry("900x680")
        self.root.minsize(800, 600)
        self.root.configure(bg=BG_DARK)
        self._center_window(900, 680)

        self._build_ui()

    def _center_window(self, w, h):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth()  // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        # ── Top navbar ──
        navbar = tk.Frame(self.root, bg=BG_CARD, height=56)
        navbar.pack(fill="x", side="top")
        navbar.pack_propagate(False)

        tk.Label(navbar, text="🎯  Customer Emotion Tracker",
                 font=("Arial", 14, "bold"),
                 bg=BG_CARD, fg=TEXT_WHITE).pack(side="left", padx=20)

        # User info + logout on right
        right_nav = tk.Frame(navbar, bg=BG_CARD)
        right_nav.pack(side="right", padx=20)

        tk.Label(right_nav, text=f"👤  {self.username.capitalize()}",
                 font=("Arial", 10), bg=BG_CARD, fg=TEXT_DIM).pack(side="left", padx=(0, 16))

        logout_btn = tk.Button(right_nav, text="Logout",
                                font=("Arial", 9, "bold"),
                                bg=ACCENT, fg=TEXT_WHITE,
                                activebackground=ACCENT2,
                                relief="flat", bd=0, cursor="hand2",
                                padx=12, pady=4,
                                command=self._logout)
        logout_btn.pack(side="left")

        # ── Tabs ──
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                         background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                         background=BG_CARD, foreground=TEXT_DIM,
                         padding=[20, 8], font=("Arial", 10, "bold"),
                         borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_DARK)],
                  foreground=[("selected", ACCENT)])

        notebook = ttk.Notebook(self.root, style="Dark.TNotebook")
        notebook.pack(fill="both", expand=True, padx=0, pady=0)

        # Tab 1 — Emotion Predictor
        tab1 = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab1, text="  Emotion Predictor  ")
        self._build_predictor_tab(tab1)

        # Tab 2 — ML Metrics
        tab2 = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab2, text="  ML Metrics  ")
        self._build_metrics_tab(tab2)

        # Tab 3 — About
        tab3 = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab3, text="  About  ")
        self._build_about_tab(tab3)

    # ──────────────────────────────────────────────────────────
    #  TAB 1 — EMOTION PREDICTOR
    # ──────────────────────────────────────────────────────────
    def _build_predictor_tab(self, parent):
        main = tk.Frame(parent, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=24, pady=20)

        # Left panel — input
        left = tk.Frame(main, bg=BG_CARD, bd=0,
                        highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        tk.Label(left, text="Customer Feedback",
                 font=("Arial", 13, "bold"),
                 bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w", padx=20, pady=(16, 4))

        tk.Label(left, text="Enter customer review, support message, or social media text:",
                 font=("Arial", 9), bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", padx=20)

        # Text area
        text_frame = tk.Frame(left, bg=BG_INPUT, bd=0,
                              highlightthickness=1, highlightbackground=BORDER)
        text_frame.pack(fill="both", expand=True, padx=20, pady=12)

        self.text_input = tk.Text(text_frame, font=("Arial", 12),
                                   bg=BG_INPUT, fg=TEXT_PRIMARY,
                                   insertbackground=ACCENT,
                                   relief="flat", bd=0, wrap="word",
                                   padx=12, pady=12)
        self.text_input.pack(fill="both", expand=True)

        # Placeholder
        self._add_placeholder(self.text_input,
                              "Type or paste customer feedback here...")

        # Buttons row
        btn_row = tk.Frame(left, bg=BG_CARD)
        btn_row.pack(fill="x", padx=20, pady=(0, 16))

        analyze_btn = tk.Button(btn_row, text="Analyse Emotion  →",
                                 font=("Arial", 11, "bold"),
                                 bg=ACCENT, fg=TEXT_WHITE,
                                 activebackground=ACCENT2,
                                 relief="flat", bd=0, cursor="hand2",
                                 padx=20, pady=10,
                                 command=self._analyse)
        analyze_btn.pack(side="left")
        analyze_btn.bind("<Enter>", lambda e: analyze_btn.config(bg=ACCENT2))
        analyze_btn.bind("<Leave>", lambda e: analyze_btn.config(bg=ACCENT))

        clear_btn = tk.Button(btn_row, text="Clear",
                               font=("Arial", 10),
                               bg=BG_INPUT, fg=TEXT_DIM,
                               activebackground=BORDER,
                               relief="flat", bd=0, cursor="hand2",
                               padx=16, pady=10,
                               command=self._clear)
        clear_btn.pack(side="left", padx=(12, 0))

        # Right panel — results
        right = tk.Frame(main, bg=BG_CARD, bd=0, width=300,
                         highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="Analysis Result",
                 font=("Arial", 13, "bold"),
                 bg=BG_CARD, fg=TEXT_WHITE).pack(anchor="w", padx=20, pady=(16, 12))

        # Emotion display
        self.emotion_frame = tk.Frame(right, bg=BG_DARK, bd=0,
                                       highlightthickness=1,
                                       highlightbackground=BORDER)
        self.emotion_frame.pack(fill="x", padx=20, pady=(0, 12))

        self.emoji_label = tk.Label(self.emotion_frame, text="❓",
                                     font=("Arial", 40),
                                     bg=BG_DARK, fg=TEXT_DIM)
        self.emoji_label.pack(pady=(16, 4))

        self.emotion_label = tk.Label(self.emotion_frame, text="—",
                                       font=("Arial", 22, "bold"),
                                       bg=BG_DARK, fg=TEXT_DIM)
        self.emotion_label.pack()

        self.confidence_label = tk.Label(self.emotion_frame, text="Confidence: —",
                                          font=("Arial", 10),
                                          bg=BG_DARK, fg=TEXT_DIM)
        self.confidence_label.pack(pady=(4, 16))

        # Confidence bar
        bar_frame = tk.Frame(right, bg=BG_CARD)
        bar_frame.pack(fill="x", padx=20, pady=(0, 12))
        tk.Label(bar_frame, text="Confidence",
                 font=("Arial", 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")
        self.conf_bar_bg = tk.Frame(bar_frame, bg=BG_INPUT, height=8)
        self.conf_bar_bg.pack(fill="x", pady=(4, 0))
        self.conf_bar_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(self.conf_bar_bg, bg=TEXT_DIM, height=8, width=0)
        self.conf_bar.place(x=0, y=0, relheight=1, width=0)

        # Churn risk
        churn_frame = tk.Frame(right, bg=BG_CARD)
        churn_frame.pack(fill="x", padx=20, pady=(0, 12))
        tk.Label(churn_frame, text="CHURN RISK",
                 font=("Arial", 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")
        self.churn_label = tk.Label(churn_frame, text="—",
                                     font=("Arial", 14, "bold"),
                                     bg=BG_CARD, fg=TEXT_DIM)
        self.churn_label.pack(anchor="w", pady=(4, 0))

        # Suggestion
        sugg_frame = tk.Frame(right, bg=BG_INPUT, bd=0,
                              highlightthickness=1, highlightbackground=BORDER)
        sugg_frame.pack(fill="x", padx=20, pady=(0, 16))
        tk.Label(sugg_frame, text="💡  RECOMMENDATION",
                 font=("Arial", 8, "bold"),
                 bg=BG_INPUT, fg=TEXT_DIM).pack(anchor="w", padx=12, pady=(10, 4))
        self.suggestion_label = tk.Label(sugg_frame,
                                          text="Analyse a customer message to see a recommendation.",
                                          font=("Arial", 9),
                                          bg=BG_INPUT, fg=TEXT_DIM,
                                          wraplength=240, justify="left")
        self.suggestion_label.pack(anchor="w", padx=12, pady=(0, 12))

        # Model status
        status_text = f"Model: {'✅ ML Model Loaded' if model else '⚠ Rule-based Fallback'}"
        tk.Label(right, text=status_text,
                 font=("Arial", 8), bg=BG_CARD,
                 fg=SUCCESS if model else WARNING).pack(pady=(0, 12))

    def _add_placeholder(self, widget, placeholder):
        widget.insert("1.0", placeholder)
        widget.config(fg=TEXT_DIM)

        def on_focus_in(e):
            if widget.get("1.0", "end-1c") == placeholder:
                widget.delete("1.0", "end")
                widget.config(fg=TEXT_PRIMARY)

        def on_focus_out(e):
            if not widget.get("1.0", "end-1c").strip():
                widget.insert("1.0", placeholder)
                widget.config(fg=TEXT_DIM)

        widget.bind("<FocusIn>",  on_focus_in)
        widget.bind("<FocusOut>", on_focus_out)
        self._placeholder_text = placeholder

    def _get_text(self):
        text = self.text_input.get("1.0", "end-1c").strip()
        if text == self._placeholder_text:
            return ""
        return text

    def _analyse(self):
        text = self._get_text()
        if not text:
            messagebox.showwarning("Empty Input",
                                   "Please enter some customer feedback to analyse.")
            return

        emotion, confidence = predict(text)
        if not emotion:
            return

        color  = EMOTION_COLORS.get(emotion, TEXT_DIM)
        emoji  = EMOTION_EMOJIS.get(emotion, "❓")
        churn, churn_color = CHURN_RISK.get(emotion, ("—", TEXT_DIM))
        suggestion = SUGGESTIONS.get(emotion, "")

        # Update UI
        self.emoji_label.config(text=emoji, fg=color)
        self.emotion_label.config(text=emotion, fg=color)
        self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%", fg=color)
        self.emotion_frame.config(highlightbackground=color)

        # Confidence bar
        self.conf_bar_bg.update_idletasks()
        bar_width = int(self.conf_bar_bg.winfo_width() * confidence)
        self.conf_bar.place(x=0, y=0, relheight=1, width=bar_width)
        self.conf_bar.config(bg=color)

        self.churn_label.config(text=churn, fg=churn_color)
        self.suggestion_label.config(text=suggestion, fg=TEXT_PRIMARY)

    def _clear(self):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", self._placeholder_text)
        self.text_input.config(fg=TEXT_DIM)

        self.emoji_label.config(text="❓", fg=TEXT_DIM)
        self.emotion_label.config(text="—", fg=TEXT_DIM)
        self.confidence_label.config(text="Confidence: —", fg=TEXT_DIM)
        self.emotion_frame.config(highlightbackground=BORDER)
        self.conf_bar.place(x=0, y=0, relheight=1, width=0)
        self.churn_label.config(text="—", fg=TEXT_DIM)
        self.suggestion_label.config(
            text="Analyse a customer message to see a recommendation.",
            fg=TEXT_DIM)

    # ──────────────────────────────────────────────────────────
    #  TAB 2 — ML METRICS
    # ──────────────────────────────────────────────────────────
    def _build_metrics_tab(self, parent):
        canvas = tk.Canvas(parent, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas, bg=BG_DARK)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        pad = dict(padx=24, pady=8)

        tk.Label(frame, text="Model Performance Summary",
                 font=("Arial", 15, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE).pack(anchor="w", padx=24, pady=(20, 4))

        tk.Label(frame, text="Results from train_model.py — Synthetic + Real Dataset Training",
                 font=("Arial", 9), bg=BG_DARK, fg=TEXT_DIM).pack(anchor="w", padx=24)

        # Metrics cards
        cards_frame = tk.Frame(frame, bg=BG_DARK)
        cards_frame.pack(fill="x", padx=24, pady=16)

        metrics = [
            ("Best Model",  "Naive Bayes",  ACCENT),
            ("Accuracy",    "100.00%",      SUCCESS),
            ("F1 Score",    "1.0000",       INFO),
            ("CV Score",    "99.85%",       ACCENT2),
        ]

        for i, (label, value, color) in enumerate(metrics):
            card = tk.Frame(cards_frame, bg=BG_CARD, bd=0,
                            highlightthickness=1, highlightbackground=color)
            card.grid(row=0, column=i, padx=8, pady=4, sticky="ew")
            cards_frame.columnconfigure(i, weight=1)

            tk.Label(card, text=value,
                     font=("Arial", 18, "bold"),
                     bg=BG_CARD, fg=color).pack(pady=(14, 2))
            tk.Label(card, text=label,
                     font=("Arial", 9),
                     bg=BG_CARD, fg=TEXT_DIM).pack(pady=(0, 14))

        # Model comparison table
        tk.Label(frame, text="Model Comparison",
                 font=("Arial", 13, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE).pack(anchor="w", **pad)

        table_data = [
            ("Model",                "Accuracy", "F1 Score", "CV Score"),
            ("Logistic Regression",  "98.83%",   "0.9883",   "99.85%"),
            ("Naive Bayes (Best)",   "100.00%",  "1.0000",   "99.85%"),
            ("Linear SVM",           "98.83%",   "0.9883",   "99.85%"),
            ("PyTorch MLP (Real)",   "85-93%",   "~0.88",    "N/A"),
        ]

        table_frame = tk.Frame(frame, bg=BG_CARD, bd=0,
                               highlightthickness=1, highlightbackground=BORDER)
        table_frame.pack(fill="x", padx=24, pady=(0, 8))

        cols = [240, 120, 120, 120]
        for r, row in enumerate(table_data):
            for c, val in enumerate(row):
                is_header = (r == 0)
                is_best   = "Best" in row[0]
                bg = HEADER_BG = BG_CARD if not is_header else "#0F0F2A"
                if is_best and not is_header:
                    bg = "#1A2A1A"
                fg = ACCENT if is_header else (SUCCESS if is_best else TEXT_PRIMARY)
                cell = tk.Frame(table_frame, bg=bg, width=cols[c])
                cell.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
                cell.pack_propagate(False)
                tk.Label(cell, text=val,
                         font=("Arial", 10, "bold" if is_header else "normal"),
                         bg=bg, fg=fg, anchor="w").pack(padx=12, pady=8, anchor="w")

        # Classification report
        tk.Label(frame, text="Classification Report (Naive Bayes — Synthetic)",
                 font=("Arial", 13, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE).pack(anchor="w", **pad)

        report_data = [
            ("Emotion",     "Precision", "Recall", "F1-Score", "Support"),
            ("Angry",       "1.00",      "1.00",   "1.00",     "57"),
            ("Excited",     "1.00",      "1.00",   "1.00",     "57"),
            ("Frustrated",  "1.00",      "1.00",   "1.00",     "57"),
            ("Happy",       "1.00",      "1.00",   "1.00",     "57"),
            ("Neutral",     "1.00",      "1.00",   "1.00",     "57"),
            ("Sad",         "1.00",      "1.00",   "1.00",     "57"),
            ("Accuracy",    "",          "",       "1.00",     "342"),
        ]

        rep_frame = tk.Frame(frame, bg=BG_CARD, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        rep_frame.pack(fill="x", padx=24, pady=(0, 24))

        rep_cols = [140, 100, 100, 100, 100]
        for r, row in enumerate(report_data):
            for c, val in enumerate(row):
                is_header = (r == 0)
                bg = "#0F0F2A" if is_header else BG_CARD
                fg = ACCENT if is_header else TEXT_PRIMARY
                cell = tk.Frame(rep_frame, bg=bg, width=rep_cols[c])
                cell.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
                cell.pack_propagate(False)
                tk.Label(cell, text=val,
                         font=("Arial", 10, "bold" if is_header else "normal"),
                         bg=bg, fg=fg, anchor="w").pack(padx=12, pady=7, anchor="w")

    # ──────────────────────────────────────────────────────────
    #  TAB 3 — ABOUT
    # ──────────────────────────────────────────────────────────
    def _build_about_tab(self, parent):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(fill="both", expand=True, padx=40, pady=30)

        tk.Label(frame, text="🎯  Customer Emotion Tracker",
                 font=("Arial", 20, "bold"),
                 bg=BG_DARK, fg=TEXT_WHITE).pack(pady=(0, 4))

        tk.Label(frame, text="Machine Learning Project — Review 3",
                 font=("Arial", 11, "italic"),
                 bg=BG_DARK, fg=TEXT_DIM).pack(pady=(0, 24))

        info_items = [
            ("Course",      "U24IT401 – Artificial Intelligence & Machine Learning"),
            ("Institution", "Meenakshi Sundararajan Engineering College, Chennai"),
            ("Year",        "2024 – 2025"),
            ("Team",        "Amirtha Shree L  |  Harshawardhini M G  |  Mahalakshmi P"),
            ("Models",      "Logistic Regression, Naive Bayes, Linear SVM, PyTorch MLP"),
            ("Emotions",    "Happy, Angry, Frustrated, Sad, Excited, Neutral"),
            ("Dataset",     "8 Synthetic Domains (1,710 samples) + Kaggle NLP (~20,000)"),
        ]

        for label, value in info_items:
            row = tk.Frame(frame, bg=BG_CARD, bd=0,
                           highlightthickness=1, highlightbackground=BORDER)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=f"  {label}",
                     font=("Arial", 10, "bold"), width=14, anchor="w",
                     bg=BG_CARD, fg=ACCENT).pack(side="left", padx=(8, 0), pady=10)
            tk.Label(row, text=value,
                     font=("Arial", 10),
                     bg=BG_CARD, fg=TEXT_PRIMARY, anchor="w").pack(side="left", pady=10)

    # ──────────────────────────────────────────────────────────
    #  LOGOUT
    # ──────────────────────────────────────────────────────────
    def _logout(self):
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            self.root.destroy()
            main()   # restart from login


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()

    def on_login_success(username):
        # Destroy login window and open main app
        for widget in root.winfo_children():
            widget.destroy()
        root.geometry("900x680")
        root.resizable(True, True)
        EmotionTrackerApp(root, username=username)

    LoginWindow(root, on_success=on_login_success)
    root.mainloop()


if __name__ == "__main__":
    main()