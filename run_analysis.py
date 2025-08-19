import os
import re
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def ensure_dirs() -> None:
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def scrape_tweets_window(days_back: int = 120, limit_ae: int = 4000, limit_rl: int = 2000) -> pd.DataFrame:
    import snscrape.modules.twitter as sntwitter

    today = datetime.utcnow().date()
    since = (today - timedelta(days=days_back)).isoformat()
    until = today.isoformat()

    ae_terms = '("American Eagle" OR AEO) ("Sydney Sweeney" OR ad OR campaign) lang:en'
    rl_terms = '("Ralph Lauren" OR RL) ("Oak Bluffs" OR campaign OR ad) lang:en'

    def scrape(query: str, limit: int) -> pd.DataFrame:
        rows = []
        for i, t in enumerate(sntwitter.TwitterSearchScraper(f"{query} since:{since} until:{until}").get_items()):
            if i >= limit:
                break
            rows.append({
                "date": pd.to_datetime(t.date),
                "text": t.rawContent,
                "likes": t.likeCount,
                "retweets": t.retweetCount,
                "replies": getattr(t, "replyCount", np.nan),
                "quotes": getattr(t, "quoteCount", np.nan),
                "user": t.user.username,
            })
        return pd.DataFrame(rows)

    ae = scrape(ae_terms, limit_ae)
    rl = scrape(rl_terms, limit_rl)

    ae["brand"] = "American Eagle"
    rl["brand"] = "Ralph Lauren"
    df = pd.concat([ae, rl], ignore_index=True)
    df = df.drop_duplicates(subset=["date", "text"]).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text_clean"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["date_d"] = df["date"].dt.date
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (df.groupby(["brand", "date_d"])\
               .agg(mean_sentiment=("sentiment", "mean"),
                    volume=("text", "size"),
                    engagement=("likes", "sum"))\
               .reset_index())
    return daily


def plot_sentiment_volume(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=daily, x="date_d", y="mean_sentiment", hue="brand")
    plt.title("Mean sentiment over time")
    plt.ylabel("Compound sentiment")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("figures/sentiment_over_time.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 3.5))
    sns.lineplot(data=daily, x="date_d", y="volume", hue="brand")
    plt.title("Tweet volume over time")
    plt.ylabel("Tweets")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("figures/volume_over_time.png", dpi=180, bbox_inches="tight")
    plt.close()


def overlay_stock(daily: pd.DataFrame, since: str, until: str) -> None:
    import yfinance as yf

    try:
        aeo = yf.download("AEO", start=since, end=until, progress=False)[["Adj Close"]].rename(columns={"Adj Close": "AEO"})
        rlc = yf.download("RL", start=since, end=until, progress=False)[["Adj Close"]].rename(columns={"Adj Close": "RL"})
        prices = pd.concat([aeo, rlc], axis=1)
        prices["AEO_idx"] = prices["AEO"] / prices["AEO"].iloc[0] if "AEO" in prices else np.nan
        prices["RL_idx"] = prices["RL"] / prices["RL"].iloc[0] if "RL" in prices else np.nan
        prices = prices.reset_index().rename(columns={"Date": "date_d"})
    except Exception as e:
        print("Stock download failed:", e)
        return

    if prices.empty:
        print("Skipping stock overlay: no price data")
        return

    daily_wide = daily.pivot(index="date_d", columns="brand", values="mean_sentiment").reset_index()
    merged = prices.merge(daily_wide, on="date_d", how="left")

    fig, ax = plt.subplots(figsize=(11, 5))
    if "American Eagle" in daily["brand"].unique():
        sns.lineplot(data=merged, x="date_d", y="American Eagle", ax=ax, label="Sentiment - American Eagle", color="#4C78A8")
    if "Ralph Lauren" in daily["brand"].unique():
        sns.lineplot(data=merged, x="date_d", y="Ralph Lauren", ax=ax, label="Sentiment - Ralph Lauren", color="#72B7B2")
    ax.set_ylabel("Mean sentiment")
    ax2 = ax.twinx()
    if "AEO_idx" in merged:
        sns.lineplot(data=merged, x="date_d", y="AEO_idx", ax=ax2, label="AEO stock (indexed)", color="#9C755F", alpha=0.7)
    if "RL_idx" in merged:
        sns.lineplot(data=merged, x="date_d", y="RL_idx", ax=ax2, label="RL stock (indexed)", color="#F58518", alpha=0.7)
    ax.set_title("Public sentiment vs. stock (indexed)")
    ax2.set_ylabel("Indexed price")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/sentiment_vs_stock.png", dpi=180, bbox_inches="tight")
    plt.close()


def inclusion_analysis(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "Black": ["black", "african american", "blk"],
        "Latine": ["latino", "latina", "latinx", "hispanic"],
        "Asian": ["asian", "aapi", "chinese", "korean", "japanese", "filipino", "desi", "south asian", "indian"],
        "Indigenous": ["indigenous", "native american", "navajo", "cherokee", "first nations"],
        "MENA": ["mena", "arab", "middle eastern", "egyptian", "moroccan", "lebanese", "iranian", "persian"],
        "White": ["white", "caucasian"],
        "Mixed/Other": ["mixed", "biracial", "multiracial", "immigrant", "diaspora", "diverse", "inclusive", "representation"],
    }

    def tag_groups(text: str):
        lt = (text or "").lower()
        tags = [g for g, kws in groups.items() if any(kw in lt for kw in kws)]
        return list(set(tags)) or np.nan

    exp = df.copy()
    exp["groups"] = exp["text_clean"].apply(tag_groups)
    exp = exp.explode("groups")
    by_brand_group = (
        exp.dropna(subset=["groups"]).groupby(["brand", "groups"]).agg(
            mean_sent=("sentiment", "mean"), count=("text", "size")
        ).reset_index()
    )

    by_brand_group.to_csv("data/sentiment_by_group.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.barplot(data=by_brand_group, x="groups", y="mean_sent", hue="brand")
    plt.title("Sentiment when inclusion-related groups are mentioned")
    plt.ylabel("Mean sentiment")
    plt.xlabel("Mentioned group")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("figures/inclusion_groups_sentiment.png", dpi=180, bbox_inches="tight")
    plt.close()

    return by_brand_group


def topic_modeling(df: pd.DataFrame) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    from nltk.corpus import stopwords

    stop = set(stopwords.words("english"))
    vect = TfidfVectorizer(max_df=0.8, min_df=5, stop_words=stop, ngram_range=(1, 2))
    X = vect.fit_transform(df["text_clean"]) 
    if X.shape[0] == 0 or X.shape[1] == 0:
        print("Skipping topic modeling: not enough text after cleaning")
        return

    n_components = min(6, max(2, X.shape[0] // 200))
    nmf = NMF(n_components=n_components, random_state=0, init="nndsvd").fit(X)
    terms = vect.get_feature_names_out()
    topic_top_terms = {
        i: [terms[j] for j in nmf.components_[i].argsort()[-10:][::-1]] for i in range(nmf.n_components)
    }
    W = nmf.transform(X).argmax(axis=1)
    df_topics = df.copy()
    df_topics["topic"] = W
    topic_sent = df_topics.groupby(["brand", "topic"])['sentiment'].mean().reset_index()
    topic_sent.to_csv("data/sentiment_by_topic.csv", index=False)

    print("Topics (top terms):")
    for i, words in topic_top_terms.items():
        print(i, ":", ", ".join(words))

    plt.figure(figsize=(9, 4))
    sns.barplot(data=topic_sent, x="topic", y="sentiment", hue="brand")
    plt.title("Sentiment by topic")
    plt.tight_layout()
    plt.savefig("figures/sentiment_by_topic.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()

    today = datetime.utcnow().date()
    since = (today - timedelta(days=120)).isoformat()
    until = today.isoformat()

    print("Scraping tweets...")
    df_raw = scrape_tweets_window(days_back=120)
    print(f"Tweets fetched: {len(df_raw)}")

    print("Scoring sentiment...")
    df = score_sentiment(df_raw)
    df.to_csv("data/tweets_scored.csv", index=False)

    print("Aggregating daily metrics...")
    daily = aggregate_daily(df)
    daily.to_csv("data/daily_sentiment.csv", index=False)
    plot_sentiment_volume(daily)

    print("Downloading stock and overlaying...")
    overlay_stock(daily, since, until)

    print("Analyzing inclusion-related mentions...")
    inclusion_analysis(df)

    print("Running topic modeling...")
    topic_modeling(df)

    print("Done. Figures saved under ./figures and CSVs under ./data")


if __name__ == "__main__":
    main()

