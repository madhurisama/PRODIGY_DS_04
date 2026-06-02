import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords


cols = ["TweetID", "Entity", "Sentiment", "Text"]

try:
    train = pd.read_csv("twitter_training.csv", header=None, names=cols)
    val   = pd.read_csv("twitter_validation.csv", header=None, names=cols)
    df    = pd.concat([train, val], ignore_index=True)
    print(f"Loaded training + validation. Shape: {df.shape}")
except FileNotFoundError:
    # fallback: single file
    df = pd.read_csv("twitter_training.csv", header=None, names=cols)
    print(f"Loaded training only. Shape: {df.shape}")

print(df.head())
print(f"\nSentiments : {df['Sentiment'].unique()}")
print(f"Entities   : {df['Entity'].nunique()} unique brands/topics")



df.dropna(subset=["Text", "Sentiment"], inplace=True)


df = df[df["Sentiment"].isin(["Positive", "Negative", "Neutral", "Irrelevant"])]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       
    text = re.sub(r"@\w+", "", text)                  
    text = re.sub(r"#(\w+)", r"\1", text)             
    text = re.sub(r"[^a-z\s]", "", text)              
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Clean_Text"] = df["Text"].apply(clean_text)
df["Word_Count"] = df["Clean_Text"].apply(lambda x: len(x.split()))

print(f"\nAfter cleaning: {df.shape}")
print(df["Sentiment"].value_counts())



STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update(["im", "dont", "cant", "just", "like", "get",
                   "one", "would", "also", "us", "amp"])

def get_words(sentiment):
    texts = " ".join(df[df["Sentiment"] == sentiment]["Clean_Text"])
    words = [w for w in texts.split() if w not in STOPWORDS and len(w) > 2]
    return words

pos_words  = get_words("Positive")
neg_words  = get_words("Negative")
neu_words  = get_words("Neutral")


top_entities = df["Entity"].value_counts().head(10).index.tolist()
df_top = df[df["Entity"].isin(top_entities)]



sns.set_theme(style="whitegrid")
COLORS = {
    "Positive"   : "#2ecc71",
    "Negative"   : "#e74c3c",
    "Neutral"    : "#3498db",
    "Irrelevant" : "#95a5a6"
}
palette = [COLORS[s] for s in ["Positive", "Negative", "Neutral", "Irrelevant"]]

fig = plt.figure(figsize=(22, 28))
fig.suptitle("Social Media Sentiment Analysis Dashboard",
             fontsize=24, fontweight="bold", y=1.005)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)



ax1 = fig.add_subplot(gs[0, 0])
sent_counts = df["Sentiment"].value_counts()
wedge_colors = [COLORS[s] for s in sent_counts.index]
wedges, texts, autotexts = ax1.pie(
    sent_counts.values,
    labels=sent_counts.index,
    colors=wedge_colors,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax1.set_title("Overall Sentiment Distribution", fontsize=13, fontweight="bold")



ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(sent_counts.index, sent_counts.values,
               color=wedge_colors, edgecolor="white", linewidth=0.8)
ax2.set_title("Sentiment Count", fontsize=13, fontweight="bold")
ax2.set_ylabel("Number of Tweets")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 20,
             f"{int(bar.get_height()):,}",
             ha="center", fontsize=9, fontweight="bold")



ax3 = fig.add_subplot(gs[0, 2])
pivot = (df_top.groupby(["Entity", "Sentiment"])
               .size()
               .unstack(fill_value=0))

pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_pct = pivot_pct[[c for c in ["Positive","Negative","Neutral","Irrelevant"]
                        if c in pivot_pct.columns]]
pivot_pct.plot(kind="barh", stacked=True,
               color=[COLORS[c] for c in pivot_pct.columns],
               edgecolor="white", linewidth=0.5, ax=ax3)
ax3.set_title("Sentiment % by Brand/Topic", fontsize=13, fontweight="bold")
ax3.set_xlabel("Percentage (%)")
ax3.legend(loc="lower right", fontsize=8)



ax4 = fig.add_subplot(gs[1, 0])
wc_pos = WordCloud(width=600, height=300, background_color="white",
                   colormap="Greens", max_words=80,
                   stopwords=STOPWORDS).generate(" ".join(pos_words))
ax4.imshow(wc_pos, interpolation="bilinear")
ax4.axis("off")
ax4.set_title("Word Cloud — Positive Tweets", fontsize=13,
              fontweight="bold", color="#2ecc71")



ax5 = fig.add_subplot(gs[1, 1])
wc_neg = WordCloud(width=600, height=300, background_color="white",
                   colormap="Reds", max_words=80,
                   stopwords=STOPWORDS).generate(" ".join(neg_words))
ax5.imshow(wc_neg, interpolation="bilinear")
ax5.axis("off")
ax5.set_title("Word Cloud — Negative Tweets", fontsize=13,
              fontweight="bold", color="#e74c3c")



ax6 = fig.add_subplot(gs[1, 2])
wc_neu = WordCloud(width=600, height=300, background_color="white",
                   colormap="Blues", max_words=80,
                   stopwords=STOPWORDS).generate(" ".join(neu_words))
ax6.imshow(wc_neu, interpolation="bilinear")
ax6.axis("off")
ax6.set_title("Word Cloud — Neutral Tweets", fontsize=13,
              fontweight="bold", color="#3498db")



ax7 = fig.add_subplot(gs[2, 0])
top_pos = Counter(pos_words).most_common(15)
words_p, counts_p = zip(*top_pos)
ax7.barh(words_p[::-1], counts_p[::-1], color="#2ecc71", edgecolor="white")
ax7.set_title("Top 15 Positive Words", fontsize=13, fontweight="bold")
ax7.set_xlabel("Frequency")



ax8 = fig.add_subplot(gs[2, 1])
top_neg = Counter(neg_words).most_common(15)
words_n, counts_n = zip(*top_neg)
ax8.barh(words_n[::-1], counts_n[::-1], color="#e74c3c", edgecolor="white")
ax8.set_title("Top 15 Negative Words", fontsize=13, fontweight="bold")
ax8.set_xlabel("Frequency")


# ── Plot 9: Tweet Word Count by Sentiment ──
ax9 = fig.add_subplot(gs[2, 2])
sns.boxplot(x="Sentiment", y="Word_Count",
            data=df[df["Word_Count"] < 60],
            palette=COLORS, order=["Positive","Negative","Neutral","Irrelevant"],
            ax=ax9)
ax9.set_title("Tweet Length by Sentiment", fontsize=13, fontweight="bold")
ax9.set_ylabel("Word Count")
ax9.set_xlabel("")



ax10 = fig.add_subplot(gs[3, 0])
top10 = df["Entity"].value_counts().head(10)
ax10.barh(top10.index[::-1], top10.values[::-1],
          color="#9b59b6", edgecolor="white")
ax10.set_title("Top 10 Brands by Tweet Volume", fontsize=13, fontweight="bold")
ax10.set_xlabel("Tweet Count")
for i, v in enumerate(top10.values[::-1]):
    ax10.text(v + 5, i, f"{v:,}", va="center", fontsize=8)



ax11 = fig.add_subplot(gs[3, 1])
pos_rate = (df_top.groupby("Entity")
                  .apply(lambda x: (x["Sentiment"] == "Positive").mean() * 100)
                  .sort_values())
colors_bar = ["#2ecc71" if v >= pos_rate.median() else "#e74c3c"
              for v in pos_rate.values]
ax11.barh(pos_rate.index, pos_rate.values, color=colors_bar, edgecolor="white")
ax11.axvline(pos_rate.median(), color="gray", linestyle="--",
             linewidth=1.2, label=f"Median: {pos_rate.median():.1f}%")
ax11.set_title("Positive Sentiment Rate by Brand", fontsize=13, fontweight="bold")
ax11.set_xlabel("Positive Rate (%)")
ax11.legend(fontsize=9)


ax12 = fig.add_subplot(gs[3, 2])
neg_rate = (df_top.groupby("Entity")
                  .apply(lambda x: (x["Sentiment"] == "Negative").mean() * 100)
                  .sort_values(ascending=False))
ax12.barh(neg_rate.index[::-1], neg_rate.values[::-1],
          color="#e74c3c", edgecolor="white")
ax12.set_title("Negative Sentiment Rate by Brand", fontsize=13, fontweight="bold")
ax12.set_xlabel("Negative Rate (%)")


plt.savefig("sentiment_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard saved as sentiment_analysis.png")



print("\n══════════════════════════════════")
print("         KEY FINDINGS")
print("══════════════════════════════════")
print(f"Total Tweets Analyzed : {len(df):,}")
print(f"Unique Brands/Topics  : {df['Entity'].nunique()}")
print(f"\nSentiment Breakdown:")
for s, c in df["Sentiment"].value_counts().items():
    print(f"  {s:<12}: {c:>6,} ({c/len(df)*100:.1f}%)")
print(f"\nMost Positive Brand : {pos_rate.idxmax()} ({pos_rate.max():.1f}%)")
print(f"Most Negative Brand : {neg_rate.idxmax()} ({neg_rate.max():.1f}%)")
print(f"Most Tweeted Brand  : {df['Entity'].value_counts().idxmax()}")
print(f"\nTop 5 Positive Words: {[w for w,_ in Counter(pos_words).most_common(5)]}")
print(f"Top 5 Negative Words: {[w for w,_ in Counter(neg_words).most_common(5)]}")