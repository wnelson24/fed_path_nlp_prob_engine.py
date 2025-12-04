"""
Fed-Path Probability Engine With NLP Headline Shock Detection
Author: William Nelson
Purpose: Reconstruct policy-path probabilities from OIS/FedFunds
         and map headline sentiment to repricing in terminal and 
         cut-path probability distributions.

Requirements:
- Refinitiv Eikon API
- numpy, pandas, scipy
- sklearn (for logistic regression placeholder)
- nltk or transformers (optional for stronger NLP)
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import eikon as ek
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# OPTIONAL: you can uncomment and use transformers for stronger NLP
# from transformers import pipeline

# ============================================================
# 1. CONNECT EIKON
# ============================================================

ek.set_app_key("YOUR_APP_KEY_HERE")


# ============================================================
# 2. FED FUNDS FUTURES + OIS FETCH
# ============================================================

def fetch_fed_funds_chain():
    """
    Pulls the entire FF futures chain.
    Fn: Price -> Implied Rate -> Meeting Probability Extraction
    """
    chain = "0#FF:"
    fields = ["TRDPRC_1", "BID", "ASK", "HIGH_1", "LOW_1"]
    df, err = ek.get_data(chain, fields)
    if err:
        raise ValueError(err)
    df = df.dropna(subset=["TRDPRC_1"])
    return df


def fetch_ois_curve():
    """
    Fetch short-dated OIS:
    1M, 3M, 6M, 1Y, 2Y, 5Y
    """
    instruments = ["USOIS1M=", "USOIS3M=", "USOIS6M=", 
                   "USOIS1Y=", "USOIS2Y=", "USOIS5Y="]

    fields = ["TR.Mid"]
    df, err = ek.get_data(instruments, fields)
    if err:
        raise ValueError(err)

    df = df.set_index("Instrument")
    return df["Mid"] / 100  # convert to decimals


# ============================================================
# 3. POLICY-PATH PROBABILITY EXTRACTION
# ============================================================

def implied_rate_from_future(price):
    """
    Fed Funds futures settle at: 100 - implied rate.
    """
    return (100 - price) / 100


def extract_meeting_probabilities(ff_df):
    """
    Using a simple 2-state model (0bp vs 25bp move) to extract
    implied probabilities from nearest FF future contract.
    (You can expand to multi-state if you like.)
    """
    # Take nearest contract
    row = ff_df.iloc[0]
    implied = implied_rate_from_future(row["TRDPRC_1"])

    # Set reference (example: assume current effective rate = 5.33%)
    current_rate = 5.33 / 100

    # Solve for p such that:
    # implied = p*(current+0.25%) + (1-p)*(current)
    hike_size = 0.25 / 100
    if implicit := (implied - current_rate) / hike_size:
        p_hike = max(0, min(implicit, 1))
    else:
        p_hike = 0.0

    return {
        "meeting_implied": implied,
        "prob_hike_25bp": p_hike,
        "prob_cut_25bp": 1 - p_hike
    }


def terminal_rate_from_ois(ois_curve):
    """
    Terminal rate approximation:
    Use the 1Y–2Y OIS portion as proxy for terminal pricing midpoint.
    """
    one_y = float(ois_curve["USOIS1Y="])
    two_y = float(ois_curve["USOIS2Y="])

    terminal = (one_y + two_y) / 2
    return terminal


# ============================================================
# 4. NLP HEADLINE SHOCK DETECTION
# ============================================================

def fetch_fed_related_headlines(limit=20):
    """
    Pulls recent Fed-related Reuters headlines.
    Search fields: 'Powell', 'FOMC', 'Fed', 'inflation'
    """
    news, err = ek.news_headlines(query="Fed OR Powell OR FOMC", count=limit)
    if err:
        raise ValueError(err)
    return news


def fetch_story_text(story_id):
    """
    Fetch full story text.
    """
    story, err = ek.news_story(story_id)
    if err:
        return ""
    return story


def classify_headlines(headlines):
    """
    Simple NLP classifier using TF-IDF + Logistic Regression placeholder.
    You may replace with transformer for stronger performance.

    Labels (conceptually):
    - hawkish = +1
    - dovish = -1
    """

    texts = []
    for _, row in headlines.iterrows():
        sid = row["storyId"]
        texts.append(fetch_story_text(sid))

    # Dummy labels: assume half hawkish, half dovish for now.
    # Replace with your own training dataset.
    labels = np.array([1 if i % 2 == 0 else -1 for i in range(len(texts))])

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression()
    clf.fit(X, labels)

    sentiment_scores = clf.predict_proba(X)[:, 1] - 0.5  # [-0.5, +0.5]

    return sentiment_scores.mean()


# ============================================================
# 5. SHOCK-TO-REPRICING MAPPING
# ============================================================

def map_shock_to_repricing(hawkish_score, prob_dict, terminal_rate):
    """
    Maps headline sentiment -> repricing of terminal and cut-path.
    A simple linear model which you can calibrate on history.
    """

    # Baseline
    ph = prob_dict["prob_hike_25bp"]
    pc = prob_dict["prob_cut_25bp"]

    # shock amplification
    delta = hawkish_score * 0.15  # slope to tune
    new_ph = min(max(ph + delta, 0), 1)
    new_pc = 1 - new_ph

    # terminal repricing
    d_terminal = hawkish_score * 0.10  # 10bps for strong sentiment

    return {
        "new_prob_hike": new_ph,
        "new_prob_cut": new_pc,
        "repriced_terminal": terminal_rate + d_terminal
    }


# ============================================================
# 6. INTEGRATED ENGINE
# ============================================================

def run_fed_path_engine():
    print("\n--- FED PATH PROBABILITY & NLP SHOCK ENGINE ---\n")

    # 1. Fetch data
    ff = fetch_fed_funds_chain()
    ois = fetch_ois_curve()

    # 2. Extract probabilities
    prob = extract_meeting_probabilities(ff)

    # 3. Terminal rate
    terminal = terminal_rate_from_ois(ois)

    # 4. Recent headlines + sentiment
    headlines = fetch_fed_related_headlines(limit=10)
    hawkishness = classify_headlines(headlines)

    # 5. Repricing
    mapped = map_shock_to_repricing(hawkishness, prob, terminal)

    # 6. Output
    print(f"Implied Meeting Rate: {prob['meeting_implied']:.3%}")
    print(f"Prob Hike 25bp:       {prob['prob_hike_25bp']:.1%}")
    print(f"Prob Cut 25bp:        {prob['prob_cut_25bp']:.1%}\n")

    print(f"Terminal Rate (base): {terminal:.3%}")
    print(f"Hawkishness Score:    {hawkishness:.3f}\n")

    print("--- NLP-Adjusted Repricing ---")
    print(f"Repriced Hike Prob:   {mapped['new_prob_hike']:.1%}")
    print(f"Repriced Cut Prob:    {mapped['new_prob_cut']:.1%}")
    print(f"Repriced Terminal:    {mapped['repriced_terminal']:.3%}\n")

    print("--- END ---\n")


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    run_fed_path_engine()
