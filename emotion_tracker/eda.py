"""
╔══════════════════════════════════════════════════════════════╗
║   EDA + PREPROCESSING ANALYSIS                               ║
║   Customer Emotion Tracker — U24IT401                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, re, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from datetime import datetime

# ── Load all 8 datasets ────────────────────────────────────────
DATASET_DIR = "datasets"
EMOTIONS    = ["Happy", "Angry", "Frustrated", "Sad", "Excited", "Neutral"]
COLORS      = ["#2ecc71","#e74c3c","#e67e22","#3498db","#9b59b6","#95a5a6"]
COLOR_MAP   = dict(zip(EMOTIONS, COLORS))

def load_all():
    frames = []
    for f in sorted(os.listdir(DATASET_DIR)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATASET_DIR, f))
            df["dataset"] = f.replace(".csv","")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

# ════════════════════════════════════════════════════════════════
#  STEP 1 — RAW DATA OVERVIEW
# ════════════════════════════════════════════════════════════════

def step1_overview(df):
    print("\n" + "═"*60)
    print("  STEP 1 : RAW DATA OVERVIEW")
    print("═"*60)
    print(f"  Total samples      : {len(df)}")
    print(f"  Columns            : {list(df.columns)}")
    print(f"  Datasets loaded    : {df['dataset'].nunique()}")
    print(f"  Emotion classes    : {sorted(df['emotion'].unique())}")
    print(f"  Missing values     :\n{df.isnull().sum()}")
    print(f"\n  First 5 rows:")
    print(df[["text","emotion","dataset"]].head())
    print(f"\n  Class distribution:")
    print(df["emotion"].value_counts())

# ════════════════════════════════════════════════════════════════
#  STEP 2 — TEXT PREPROCESSING
# ════════════════════════════════════════════════════════════════

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)       # remove [tags]
    text = re.sub(r"#\w+", " ", text)           # remove hashtags
    text = re.sub(r"@\w+", " ", text)           # remove @handles
    text = re.sub(r"http\S+", " ", text)        # remove URLs
    text = re.sub(r"[^\w\s]", " ", text)        # remove punctuation
    text = re.sub(r"\d+", "", text)             # remove digits
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return text

def step2_preprocess(df):
    print("\n" + "═"*60)
    print("  STEP 2 : TEXT PREPROCESSING")
    print("═"*60)

    df["text_clean"]  = df["text"].apply(preprocess_text)
    df["word_count"]  = df["text_clean"].apply(lambda x: len(x.split()))
    df["char_count"]  = df["text_clean"].apply(len)
    df["unique_words"]= df["text_clean"].apply(lambda x: len(set(x.split())))

    print("  Sample before → after preprocessing:")
    for i in [0, 50, 100]:
        print(f"\n  [{i}] BEFORE : {df['text'].iloc[i][:80]}")
        print(f"       AFTER  : {df['text_clean'].iloc[i][:80]}")

    print(f"\n  Word count stats:")
    print(df["word_count"].describe().round(2))
    return df

# ════════════════════════════════════════════════════════════════
#  STEP 3 — EDA PLOTS
# ════════════════════════════════════════════════════════════════

def step3_eda(df):
    print("\n" + "═"*60)
    print("  STEP 3 : EXPLORATORY DATA ANALYSIS (EDA)")
    print("═"*60)

    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.4)

    def ax_style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.set_title(title, color="#a78bfa", fontsize=11, fontweight="bold", pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d2d44")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")

    # ── Plot 1 : Emotion Distribution (Bar) ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["emotion"].value_counts().reindex(EMOTIONS)
    bars = ax1.bar(counts.index, counts.values,
                   color=[COLOR_MAP[e] for e in counts.index], edgecolor="none", width=0.6)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha="center", va="bottom", color="white", fontsize=9)
    ax_style(ax1, "📊 Emotion Class Distribution")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=30)

    # ── Plot 2 : Emotion Distribution (Pie) ──────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=counts.index,
        colors=[COLOR_MAP[e] for e in counts.index],
        autopct="%1.1f%%", startangle=140,
        textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)
    ax2.set_facecolor("#0f0f1a")
    ax2.set_title("🥧 Emotion Share (%)", color="#a78bfa",
                   fontsize=11, fontweight="bold")

    # ── Plot 3 : Samples per Dataset ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ds_counts = df["dataset"].value_counts()
    ds_labels = [d.split("_")[0] for d in ds_counts.index]
    ds_colors = ["#7c3aed","#2ecc71","#e74c3c","#e67e22",
                  "#3498db","#9b59b6","#f39c12","#1abc9c"]
    bars3 = ax3.barh(ds_labels, ds_counts.values,
                     color=ds_colors[:len(ds_labels)], edgecolor="none")
    for bar, val in zip(bars3, ds_counts.values):
        ax3.text(val + 2, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", color="white", fontsize=9)
    ax_style(ax3, "📁 Samples per Dataset")
    ax3.set_xlabel("Count")

    # ── Plot 4 : Word Count Distribution ─────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df["word_count"], bins=30, color="#7c3aed",
             edgecolor="#0f0f1a", alpha=0.85)
    ax4.axvline(df["word_count"].mean(), color="#f39c12",
                linestyle="--", linewidth=1.5, label=f"Mean={df['word_count'].mean():.1f}")
    ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_style(ax4, "📝 Word Count Distribution")
    ax4.set_xlabel("Words per Review")
    ax4.set_ylabel("Frequency")

    # ── Plot 5 : Avg Word Count per Emotion ──────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    avg_words = df.groupby("emotion")["word_count"].mean().reindex(EMOTIONS)
    bars5 = ax5.bar(avg_words.index, avg_words.values,
                    color=[COLOR_MAP[e] for e in avg_words.index],
                    edgecolor="none", width=0.6)
    for bar, val in zip(bars5, avg_words.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    ax_style(ax5, "📏 Avg Word Count per Emotion")
    ax5.set_ylabel("Avg Words")
    ax5.tick_params(axis="x", rotation=30)

    # ── Plot 6 : Char Count Boxplot per Emotion ───────────────
    ax6 = fig.add_subplot(gs[1, 2])
    data_by_emotion = [df[df["emotion"]==e]["char_count"].values for e in EMOTIONS]
    bp = ax6.boxplot(data_by_emotion, labels=EMOTIONS, patch_artist=True,
                     medianprops={"color":"white","linewidth":2},
                     whiskerprops={"color":"#94a3b8"},
                     capprops={"color":"#94a3b8"},
                     flierprops={"marker":"o","markersize":3,"alpha":0.5})
    for patch, col in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax_style(ax6, "📦 Char Count Boxplot by Emotion")
    ax6.set_ylabel("Characters")
    ax6.tick_params(axis="x", rotation=30)

    # ── Plot 7 : Top 15 Words — Happy ────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    _plot_top_words(ax7, df, "Happy", "#2ecc71", "😊 Top Words — Happy")

    # ── Plot 8 : Top 15 Words — Angry ────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    _plot_top_words(ax8, df, "Angry", "#e74c3c", "😡 Top Words — Angry")

    # ── Plot 9 : Top 15 Words — Frustrated ───────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    _plot_top_words(ax9, df, "Frustrated", "#e67e22", "😤 Top Words — Frustrated")

    # ── Plot 10 : Top 15 Words — Excited ─────────────────────
    ax10 = fig.add_subplot(gs[3, 0])
    _plot_top_words(ax10, df, "Excited", "#9b59b6", "🤩 Top Words — Excited")

    # ── Plot 11 : Heatmap — Emotion vs Dataset ────────────────
    ax11 = fig.add_subplot(gs[3, 1])
    pivot = df.pivot_table(index="emotion", columns="dataset",
                           aggfunc="size", fill_value=0)
    pivot.columns = [c.split("_")[0] for c in pivot.columns]
    im = ax11.imshow(pivot.values, cmap="plasma", aspect="auto")
    ax11.set_xticks(range(len(pivot.columns)))
    ax11.set_xticklabels(pivot.columns, rotation=45, ha="right",
                          color="#94a3b8", fontsize=8)
    ax11.set_yticks(range(len(pivot.index)))
    ax11.set_yticklabels(pivot.index, color="#94a3b8", fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax11.text(j, i, str(pivot.values[i,j]),
                      ha="center", va="center", color="white", fontsize=7)
    ax11.set_facecolor("#1a1a2e")
    ax11.set_title("🗺️ Emotion × Dataset Heatmap",
                    color="#a78bfa", fontsize=11, fontweight="bold")

    # ── Plot 12 : Unique Words per Emotion ────────────────────
    ax12 = fig.add_subplot(gs[3, 2])
    avg_uniq = df.groupby("emotion")["unique_words"].mean().reindex(EMOTIONS)
    bars12 = ax12.bar(avg_uniq.index, avg_uniq.values,
                      color=[COLOR_MAP[e] for e in avg_uniq.index],
                      edgecolor="none", width=0.6)
    for bar, val in zip(bars12, avg_uniq.values):
        ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    ax_style(ax12, "🔤 Avg Unique Words per Emotion")
    ax12.set_ylabel("Avg Unique Words")
    ax12.tick_params(axis="x", rotation=30)

    fig.suptitle("Customer Emotion Tracker — EDA Report\nU24IT401 · Meenakshi Sundararajan Engineering College",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig("EDA_Report.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f1a")
    print("  ✅ EDA chart saved → EDA_Report.png")
    plt.show()


def _plot_top_words(ax, df, emotion, color, title):
    texts  = " ".join(df[df["emotion"]==emotion]["text_clean"].tolist())
    stopwords = {"the","a","an","is","it","in","to","and","of","for",
                 "this","that","was","i","my","me","we","be","with",
                 "on","at","by","as","are","have","had","has","but",
                 "so","not","its","or","from","they","their","your",
                 "our","you","he","she","do","did","been","will","can"}
    words  = [w for w in texts.split() if w not in stopwords and len(w) > 2]
    top15  = Counter(words).most_common(15)
    if not top15:
        return
    labels = [w for w,_ in top15]
    vals   = [c for _,c in top15]
    bars   = ax.barh(labels[::-1], vals[::-1], color=color,
                     edgecolor="none", alpha=0.85)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_title(title, color="#a78bfa", fontsize=10, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2d44")


# ════════════════════════════════════════════════════════════════
#  STEP 4 — PREPROCESSING SUMMARY REPORT
# ════════════════════════════════════════════════════════════════

def step4_summary(df):
    print("\n" + "═"*60)
    print("  STEP 4 : PREPROCESSING SUMMARY")
    print("═"*60)

    total_words  = df["word_count"].sum()
    vocab        = set(" ".join(df["text_clean"].tolist()).split())

    print(f"  Total samples after cleaning : {len(df)}")
    print(f"  Total words in corpus        : {total_words:,}")
    print(f"  Unique vocabulary size       : {len(vocab):,}")
    print(f"  Avg words per review         : {df['word_count'].mean():.2f}")
    print(f"  Avg chars per review         : {df['char_count'].mean():.2f}")
    print(f"  Min word count               : {df['word_count'].min()}")
    print(f"  Max word count               : {df['word_count'].max()}")
    print(f"\n  Samples per emotion:")
    print(df["emotion"].value_counts().to_string())
    print(f"\n  Samples per dataset:")
    print(df["dataset"].value_counts().to_string())
    print(f"\n  ✅ Preprocessing complete. Dataset is balanced and clean.")
    print(f"  ✅ EDA saved as EDA_Report.png in your project folder.")
    print("═"*60 + "\n")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Loading datasets...")
    df = load_all()

    step1_overview(df)
    df = step2_preprocess(df)
    step3_eda(df)
    step4_summary(df)