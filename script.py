"""
Fed-Path Probability Engine With NLP Headline Shock Detection
Workspace RDP Version (rd.open_session)
Author: William Nelson
Purpose:
    Reconstruct policy-path probabilities from OIS/FedFunds
    and map headline sentiment to repricing in terminal and
    cut-path probability distributions.
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import rd as rdp    # Workspace RDP client (no AppKey required)


# ============================================================
# 1. CONNECT WORKSPACE
# ============================================================

# This authenticates using your Refinitiv Workspace desktop login
rdp.open_session()


# ============================================================
# 2. FED FUNDS FUTURES + OIS FETCH
# ============================================================

def fetch_fed_funds_chain():
    """
    Pulls the FF futures chain (0#FF:).
    Returns last prices and implied rates.
    """
    chain = "0#FF:"
    fields = ["TRDPRC_1", "BID", "ASK", "HIGH_1", "LOW_1"]

    df = rdp.get_data(universe=[chain], fields=fields).data.df
    df = df.dropna(subset=["TRDPRC_1"])

    return df


def fetch_ois_curve():
    """
    Fetch OIS curve points:
    1M, 3M, 6M, 1Y, 2Y, 5Y
    """

    instruments = ["USOIS1M=", "USOIS3M=", "USOIS6M=",
                   "USOIS1Y=", "USOIS2Y=", "USOIS5Y="]

    df = rdp.get_data(universe=instruments, fields=["TR.Mid"]).data.df
    df = df.set_index("Instrument")

    # Convert from % to decimal rate
    return df["TR.Mid"] / 100


# ============================================================
# 3. POLICY-PATH PROBABILITY EXTRACTION
# ============================================================

def implied_rate_from_future(price):
    """
    Fed Funds futures settle at 100 - implied rate.
    """
    return (100 - price) / 100


def extract_meeting_probabilities(ff_df):
    """
    A simple 2-state model:
        0bp move vs +25bp move.
    """
    nearest = ff_df.iloc[0]
    implied = implied_rate_from_future(nearest["TRDPRC_1"])

    # Example effective rate assumption:
    current_rate = 5.33 / 100
    hike_size = 0.25 / 100

    raw = (implied - current_rate) / hike_size
    p_hike = max(0, min(raw, 1)) if hike_size != 0 else 0

    return {
        "meeting_implied": implied,
        "prob_hike_25bp": p_hike,
        "prob_cut_25bp": 1 - p_hike
    }


def terminal_rate_from_ois(ois_curve):
    """
    Terminal rate ~ midpoint of 1Y and 2Y OIS.
    """
    one_y = float(ois_curve["USOIS1Y="])
    two_y = float(ois_curve["USOIS2Y="])

    return (one_y + two_y) / 2


# ============================================================
# 4. NLP HEADLINE SHOCK DETECTION
# ============================================================

def fetch_fed_related_headlines(limit=20):
    """
    Pull Reuters Fed-related headlines via RDP.
    """
    news = rdp.get_news_headlines(query="Fed OR Powell OR FOMC", count=limit).data.df
    return news


def fetch_story_text(story_id):
    """
    Fetch full Reuters story text via RDP.
    """
    try:
        return rdp.get_news_story(story_id).data["story"]
    except Exception:
        return ""


def classify_headlines(headlines):
    """
    Simple TF-IDF + Logistic Regression placeholder.
    Produces a sentiment score ∈ [-0.5, +0.5].
    """

    texts = [fetch_story_text(sid) for sid in headlines["storyId"]]

    # Dummy labels (replace with real dataset later)
    labels = np.array([1 if i % 2 == 0 else -1 for i in range(len(texts))])

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression()
    clf.fit(X, labels)

    sentiment = clf.predict_proba(X)[:, 1] - 0.5
    return sentiment.mean()


# ============================================================
# 5. SHOCK-TO-REPRICING MAPPING
# ============================================================

def map_shock_to_repricing(hawkish_score, prob_dict, terminal_rate):
    """
    Maps the NLP hawkish/dovish sentiment → repriced probabilities & terminal rate.
    """

    ph = prob_dict["prob_hike_25bp"]

    # Amplifier (tunables)
    dP = hawkish_score * 0.15
    dT = hawkish_score * 0.10   # 10bp per 1.0 hawkish score

    new_ph = np.clip(ph + dP, 0, 1)
    new_pc = 1 - new_ph

    return {
        "new_prob_hike": new_ph,
        "new_prob_cut": new_pc,
        "repriced_terminal": terminal_rate + dT
    }


# ============================================================
# 6. INTEGRATED FED-PATH ENGINE
# ============================================================

def run_fed_path_engine():

    print("\n--- FED PATH PROBABILITY & NLP SHOCK ENGINE (Workspace RDP Version) ---\n")

    # 1. Data
    ff = fetch_fed_funds_chain()
    ois = fetch_ois_curve()

    # 2. Probabilities
    prob = extract_meeting_probabilities(ff)

    # 3. Terminal rate
    terminal = terminal_rate_from_ois(ois)

    # 4. NLP sentiment from Reuters headlines
    headlines = fetch_fed_related_headlines(limit=10)
    hawkishness = classify_headlines(headlines)

    # 5. Repricing
    adj = map_shock_to_repricing(hawkishness, prob, terminal)

    # ----------------------
    # OUTPUT
    # ----------------------
    print(f"Implied Meeting Rate: {prob['meeting_implied']:.3%}")
    print(f"Prob Hike 25bp:       {prob['prob_hike_25bp']:.1%}")
    print(f"Prob Cut 25bp:        {prob['prob_cut_25bp']:.1%}\n")

    print(f"Terminal Rate (base): {terminal:.3%}")
    print(f"Hawkishness Score:    {hawkishness:.3f}\n")

    print("--- NLP-Adjusted Repricing ---")
    print(f"Repriced Hike Prob:   {adj['new_prob_hike']:.1%}")
    print(f"Repriced Cut Prob:    {adj['new_prob_cut']:.1%}")
    print(f"Repriced Terminal:    {adj['repriced_terminal']:.3%}\n")

    print("--- END ---\n")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    run_fed_path_engine()
