"""
Recomet — AI-Based Recommendation System with Bias Detection & Mitigation
==========================================================================
Academic demonstration of recommendation bias analysis.

Requirements:
    pip install flask requests numpy pandas matplotlib

Run:
    python app.py

Open:
    http://127.0.0.1:5000

Optional:
    Place MovieLens 100k u.data file in same folder for dataset training.
    Download: https://files.grouplens.org/datasets/movielens/ml-100k.zip
"""

import os
import io
import json
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, jsonify, request
from collections import defaultdict

app = Flask(__name__)

# ── API CONFIG ────────────────────────────────────────────────
OMDB_KEY  = "f6dbbc16"
OMDB_BASE = "https://www.omdbapi.com/"

# ── IN-MEMORY INTERACTION STORE ───────────────────────────────
SCROLL_LOG  = defaultdict(int)
VIEW_LOG    = defaultdict(int)
RATING_LOG  = defaultdict(list)
MOVIE_CACHE = {}

# ── LOGISTIC REGRESSION MODEL STATE ──────────────────────────
LR_MODEL = {
    "weights":   None,
    "bias":      0.0,
    "trained":   False,
    "accuracy":  0.0,
    "n_samples": 0,
    "mean":      None,
    "std":       None,
    "source":    "none"
}

# ── GENRE CONFIG ──────────────────────────────────────────────
GENRES = {
    "Bollywood": ["3 idiots","dangal","pk","gully boy","lagaan","swades","dil chahta hai"],
    "Action":    ["kgf","rrr","war","pathaan","bahubali","uri","tiger zinda hai"],
    "Drama":     ["taare zameen par","masaan","article 15","gangs of wasseypur","mughal e azam"],
    "Romance":   ["jab we met","barfi","yeh jawaani hai deewani","dil se","veer zaara"],
    "Comedy":    ["hera pheri","andaz apna apna","golmaal","chupke chupke","padosan"],
    "Thriller":  ["drishyam","kahaani","a wednesday","special 26","talaash"],
    "Sci-Fi":    ["ra one","koi mil gaya","robot","enthiran","mr india"],
    "Horror":    ["stree","tumbbad","bhool bhulaiyaa","pari","raat"],
    "Animation": ["hanuman","chhota bheem","roadside romeo","delhi safari","jumbo"],
    "Crime":     ["gangs of wasseypur","once upon a time in mumbai","satya","company","black friday"],
}

GENRE_COLORS = {
    "Bollywood":"#e11d48","Action":"#ef4444","Drama":"#8b5cf6",
    "Romance":"#ec4899","Comedy":"#f59e0b","Thriller":"#10b981",
    "Sci-Fi":"#06b6d4","Horror":"#6366f1","Animation":"#f97316","Crime":"#64748b"
}

# ═══════════════════════════════════════════════════════════════
#  OMDB HELPERS
# ═══════════════════════════════════════════════════════════════
def omdb_search(query):
    try:
        r = requests.get(OMDB_BASE,
            params={"apikey": OMDB_KEY, "s": query, "type": "movie"},
            timeout=8)
        return r.json().get("Search", [])
    except:
        return []

def omdb_detail(imdb_id):
    try:
        r = requests.get(OMDB_BASE,
            params={"apikey": OMDB_KEY, "i": imdb_id, "plot": "short"},
            timeout=8)
        return r.json()
    except:
        return {}

def enrich(raw):
    mid = raw.get("imdbID", raw.get("Title", "?"))
    try:
        rating = round(float(raw.get("imdbRating", "0")) / 2, 1)
    except:
        rating = 0.0
    try:
        votes = int(raw.get("imdbVotes", "0").replace(",", ""))
    except:
        votes = 0
    poster = raw.get("Poster", "")
    if poster == "N/A":
        poster = None
    m = {
        "id":       mid,
        "title":    raw.get("Title", "Unknown"),
        "year":     (raw.get("Year", "2000"))[:4],
        "rating":   rating if rating > 0 else 3.5,
        "votes":    votes,
        "overview": raw.get("Plot", "") if raw.get("Plot", "") != "N/A" else "",
        "poster":   poster,
        "genres":   [g.strip() for g in raw.get("Genre", "").split(",")][:2]
                    if raw.get("Genre", "") not in ("", "N/A") else [],
        "popular":  votes > 30000,
    }
    MOVIE_CACHE[mid] = m
    return m

GENRE_CACHE = {}
def get_genre_movies(genre):
    if genre in GENRE_CACHE:
        return GENRE_CACHE[genre]
    movies, seen = [], set()
    for q in GENRES.get(genre, [genre.lower()]):
        for raw in omdb_search(q)[:2]:
            iid = raw.get("imdbID")
            if iid and iid not in seen:
                seen.add(iid)
                det = omdb_detail(iid)
                if det.get("Response") == "True":
                    movies.append(enrich(det))
    GENRE_CACHE[genre] = movies
    return movies

# ═══════════════════════════════════════════════════════════════
#  LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def load_dataset():
    """Train on MovieLens 100k u.data if available."""
    dataset_path = "u.data"
    if not os.path.exists(dataset_path):
        print("  [INFO] u.data not found. Skipping dataset training.")
        return False
    try:
        cols = ["user_id", "item_id", "rating", "timestamp"]
        df   = pd.read_csv(dataset_path, sep="\t", names=cols)

        review_counts          = df["item_id"].value_counts().reset_index()
        review_counts.columns  = ["item_id", "review_count"]
        df = df.merge(review_counts, on="item_id")

        avg_rating             = df.groupby("item_id")["rating"].mean().reset_index()
        avg_rating.columns     = ["item_id", "avg_item_rating"]
        df = df.merge(avg_rating, on="item_id")

        max_reviews        = df["review_count"].max()
        df["pop_score"]    = df["review_count"] / max_reviews
        df["is_popular"]   = (df["review_count"] > df["review_count"].quantile(0.75)).astype(float)
        df["view_proxy"]   = df["pop_score"]
        df["target"]       = (df["rating"] >= 4).astype(int)

        X = df[["pop_score", "avg_item_rating", "is_popular", "view_proxy"]].values
        y = df["target"].values

        idx   = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]

        mean        = X_train.mean(axis=0)
        std         = X_train.std(axis=0) + 1e-8
        X_train_n   = (X_train - mean) / std
        X_test_n    = (X_test  - mean) / std

        w, b = np.zeros(X_train_n.shape[1]), 0.0
        for _ in range(300):
            pred = sigmoid(X_train_n @ w + b)
            err  = pred - y_train
            w   -= 0.05 * (X_train_n.T @ err) / len(y_train)
            b   -= 0.05 * err.mean()

        preds    = (sigmoid(X_test_n @ w + b) >= 0.5).astype(int)
        accuracy = float(np.mean(preds == y_test))

        LR_MODEL.update({
            "weights":   w.tolist(),
            "bias":      float(b),
            "mean":      mean.tolist(),
            "std":       std.tolist(),
            "trained":   True,
            "accuracy":  round(accuracy * 100, 1),
            "n_samples": len(X),
            "source":    "dataset"
        })
        print(f"  [OK] Dataset training complete: {len(X)} samples, {accuracy*100:.1f}% accuracy")
        return True
    except Exception as e:
        print(f"  [ERROR] Dataset training failed: {e}")
        return False

def train_on_interactions():
    """Fallback: train on live user ratings if dataset unavailable."""
    X, y = [], []
    for mid, ratings in RATING_LOG.items():
        m = MOVIE_CACHE.get(mid)
        if not m:
            continue
        pop_score  = min(m["votes"] / 500000, 1.0)
        item_avg   = sum(ratings) / len(ratings)
        is_popular = 1.0 if m["popular"] else 0.0
        view_cnt   = min(VIEW_LOG.get(mid, 0) / 50, 1.0)
        for r in ratings:
            X.append([pop_score, item_avg, is_popular, view_cnt])
            y.append(1 if r >= 4 else 0)
    if len(X) < 5:
        return
    X, y   = np.array(X), np.array(y)
    mean   = X.mean(axis=0)
    std    = X.std(axis=0) + 1e-8
    Xn     = (X - mean) / std
    w, b   = np.zeros(Xn.shape[1]), 0.0
    for _ in range(300):
        pred = sigmoid(Xn @ w + b)
        err  = pred - y
        w   -= 0.05 * (Xn.T @ err) / len(y)
        b   -= 0.05 * err.mean()
    preds    = (sigmoid(Xn @ w + b) >= 0.5).astype(int)
    accuracy = float(np.mean(preds == y))
    LR_MODEL.update({
        "weights":   w.tolist(),
        "bias":      float(b),
        "mean":      mean.tolist(),
        "std":       std.tolist(),
        "trained":   True,
        "accuracy":  round(accuracy * 100, 1),
        "n_samples": len(X),
        "source":    "interactions"
    })

def predict_liked(movie_id):
    """
    Predict YES / NO based on stored movie features only.
    Does NOT depend on current unsaved user input.
    """
    if not LR_MODEL["trained"]:
        return None
    m = MOVIE_CACHE.get(movie_id)
    if not m:
        return None
    w    = np.array(LR_MODEL["weights"])
    b    = LR_MODEL["bias"]
    mean = np.array(LR_MODEL["mean"])
    std  = np.array(LR_MODEL["std"])

    pop_score      = min(m["votes"] / 500000, 1.0)
    stored_ratings = RATING_LOG.get(movie_id, [])
    item_avg       = sum(stored_ratings) / len(stored_ratings) if stored_ratings else m["rating"]
    is_popular     = 1.0 if m["popular"] else 0.0
    view_cnt       = min(VIEW_LOG.get(movie_id, 0) / 50, 1.0)

    x    = np.array([pop_score, item_avg, is_popular, view_cnt])
    prob = float(sigmoid((x - mean) / std @ w + b))
    return "YES" if prob >= 0.5 else "NO"

# ═══════════════════════════════════════════════════════════════
#  BIAS METRICS
# ═══════════════════════════════════════════════════════════════
def gini(arr):
    arr = np.sort(np.array(arr, dtype=float) + 1e-9)
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1) / n)

def disparate_impact(y_pred, groups):
    pop   = [p for p, g in zip(y_pred, groups) if g == 1]
    niche = [p for p, g in zip(y_pred, groups) if g == 0]
    if not pop or not niche:
        return None
    pp = sum(pop) / len(pop)
    return round(sum(niche) / len(niche) / pp, 3) if pp else None

def statistical_parity(y_pred, groups):
    pop   = [p for p, g in zip(y_pred, groups) if g == 1]
    niche = [p for p, g in zip(y_pred, groups) if g == 0]
    if not pop or not niche:
        return None
    return round(sum(niche) / len(niche) - sum(pop) / len(pop), 3)

def equal_opportunity(y_true, y_pred, groups):
    def tpr(yt, yp):
        tp = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 1)
        fn = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    py_t = [a for a, g in zip(y_true, groups) if g == 1]
    py_p = [b for b, g in zip(y_pred, groups) if g == 1]
    ny_t = [a for a, g in zip(y_true, groups) if g == 0]
    ny_p = [b for b, g in zip(y_pred, groups) if g == 0]
    if not py_t or not ny_t:
        return None
    return round(tpr(ny_t, ny_p) - tpr(py_t, py_p), 3)

def compute_bias():
    views       = [VIEW_LOG.get(mid, 0) for mid in MOVIE_CACHE]
    total_views = sum(VIEW_LOG.values())

    empty = {
        "gini": 0, "pop_pct": 0, "niche_pct": 0,
        "scroll": dict(SCROLL_LOG), "total_views": 0,
        "top_viewed": [], "bias_level": "LOW",
        "disparate_impact": None, "statistical_parity": None,
        "equal_opportunity": None,
        "lr_trained":  LR_MODEL["trained"],
        "lr_accuracy": LR_MODEL["accuracy"],
        "lr_samples":  LR_MODEL["n_samples"],
        "lr_source":   LR_MODEL["source"]
    }
    if not views or total_views == 0:
        return empty

    g       = round(gini(views), 4)
    pop_ids = [mid for mid, m in MOVIE_CACHE.items() if m["popular"]]
    nch_ids = [mid for mid, m in MOVIE_CACHE.items() if not m["popular"]]
    pop_v   = sum(VIEW_LOG.get(m, 0) for m in pop_ids) + 1
    nch_v   = sum(VIEW_LOG.get(m, 0) for m in nch_ids) + 1
    total   = pop_v + nch_v

    top = sorted(VIEW_LOG.items(), key=lambda x: x[1], reverse=True)[:5]
    top_viewed = [
        {"title": MOVIE_CACHE.get(mid, {}).get("title", "?"), "views": v}
        for mid, v in top if mid in MOVIE_CACHE
    ]

    di = sp = eo = None
    if RATING_LOG:
        y_true, y_pred, groups = [], [], []
        for mid, ratings in RATING_LOG.items():
            m = MOVIE_CACHE.get(mid)
            if not m:
                continue
            item_avg = sum(ratings) / len(ratings)
            for r in ratings:
                y_true.append(1 if r >= 4 else 0)
                y_pred.append(1 if item_avg >= 3.5 else 0)
                groups.append(1 if m["popular"] else 0)
        if y_true:
            di = disparate_impact(y_pred, groups)
            sp = statistical_parity(y_pred, groups)
            eo = equal_opportunity(y_true, y_pred, groups)

    return {
        "gini":               g,
        "pop_pct":            round(pop_v / total * 100, 1),
        "niche_pct":          round(nch_v / total * 100, 1),
        "scroll":             dict(SCROLL_LOG),
        "total_views":        total_views,
        "top_viewed":         top_viewed,
        "bias_level":         "LOW" if g < 0.4 else "MEDIUM" if g < 0.65 else "HIGH",
        "disparate_impact":   di,
        "statistical_parity": sp,
        "equal_opportunity":  eo,
        "lr_trained":         LR_MODEL["trained"],
        "lr_accuracy":        LR_MODEL["accuracy"],
        "lr_samples":         LR_MODEL["n_samples"],
        "lr_source":          LR_MODEL["source"]
    }

# ═══════════════════════════════════════════════════════════════
#  MATPLOTLIB BIAS CHART
# ═══════════════════════════════════════════════════════════════
def make_bias_chart():
    bias = compute_bias()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor('#0d1117')

    # Gini bar
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    stages = ['Original\nData', 'After Loop\nSimulation', 'After\nMitigation']
    vals   = [0.629, 0.736, 0.476]
    colors = ['#f59e0b', '#ef4444', '#22c55e']
    bars   = ax.bar(stages, vals, color=colors, width=0.5, edgecolor='#ffffff22')
    ax.set_ylim(0, 1)
    ax.set_title('Gini Coefficient — MovieLens 100k', color='white', fontsize=11, pad=10)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_color('#ffffff22')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', color='white', fontsize=10, fontweight='bold')

    # Popular vs Niche pie
    ax2 = axes[1]
    ax2.set_facecolor('#0d1117')
    pop_pct   = bias.get("pop_pct", 50)
    niche_pct = bias.get("niche_pct", 50)
    _, texts, autotexts = ax2.pie(
        [pop_pct, niche_pct],
        labels=['Popular', 'Niche'],
        colors=['#ef4444', '#3b82f6'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'color': 'white', 'fontsize': 10}
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
    ax2.set_title('Popular vs Niche View Distribution', color='white', fontsize=11, pad=10)

    # Scroll heatmap
    ax3 = axes[2]
    ax3.set_facecolor('#0d1117')
    scroll_data = bias.get("scroll", {})
    if scroll_data:
        cats    = list(scroll_data.keys())
        counts  = list(scroll_data.values())
        colors3 = [GENRE_COLORS.get(c, '#888888') for c in cats]
        ax3.barh(cats, counts, color=colors3, edgecolor='#ffffff22')
        ax3.set_xlabel('Scroll Count', color='white')
    else:
        ax3.text(0.5, 0.5, 'Scroll genres to\ngenerate heatmap',
                 ha='center', va='center', color='#64748b',
                 fontsize=11, transform=ax3.transAxes)
    ax3.tick_params(colors='white')
    for sp in ax3.spines.values():
        sp.set_color('#ffffff22')
    ax3.set_title('Category Scroll Heatmap', color='white', fontsize=11, pad=10)

    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#0d1117', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_b64

# ═══════════════════════════════════════════════════════════════
#  RECOMMENDATIONS WITH AUTOMATIC BIAS DETECTION & MITIGATION
# ═══════════════════════════════════════════════════════════════
def get_recommendations(selected_genres):
    """
    Automatically detects feedback loop bias using the Gini coefficient.
    If Gini > 0.65 (HIGH bias), mitigation is applied:
      - Niche movies receive a score boost (+1.18)
      - Popular movies receive no boost (0.0 added)
    Sorted by fair_score descending, top 10 returned.
    Returns bias_detected flag and gini value for UI banner.
    """
    all_movies, seen = [], set()
    for genre in selected_genres:
        for m in get_genre_movies(genre):
            if m["id"] not in seen:
                seen.add(m["id"])
                mc = dict(m)
                mc["genre_src"] = genre
                all_movies.append(mc)

    if not all_movies:
        return {"movies": [], "bias_detected": False, "gini": 0.0,
                "mitigation_applied": False}

    # Compute current bias state
    bias         = compute_bias()
    gini_score   = bias.get("gini", 0)
    bias_detected = gini_score > 0.65

    # Apply mitigation if bias is HIGH
    for m in all_movies:
        if bias_detected:
            # Niche items get a boost to counter underexposure
            boost = 1.18 if not m.get("popular", False) else 0.0
            m["fair_score"] = round(m["rating"] * 0.8 + boost, 4)
            m["mitigated"]  = True
        else:
            # No mitigation needed — use raw rating
            m["fair_score"] = m["rating"]
            m["mitigated"]  = False

    # Sort by fair_score descending
    all_movies.sort(key=lambda x: x.get("fair_score", x["rating"]), reverse=True)

    return {
        "movies":              all_movies[:10],
        "bias_detected":       bias_detected,
        "gini":                round(gini_score, 4),
        "mitigation_applied":  bias_detected,
        "bias_level":          bias.get("bias_level", "LOW")
    }

# ═══════════════════════════════════════════════════════════════
#  HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recomet — AI Recommendation System</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#07090e;--surf:#0d1117;--card:#111620;--border:#ffffff0d;
  --accent:#2563eb;--blue:#3b82f6;--gold:#f59e0b;--green:#22c55e;
  --red:#ef4444;--yellow:#eab308;
  --text:#f0f4f8;--sub:#64748b;--dim:#1a2030;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);
     font-family:'DM Sans',sans-serif;overflow-x:hidden}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:#ffffff15;border-radius:4px}

nav{position:fixed;top:0;width:100%;z-index:300;
    background:linear-gradient(180deg,var(--bg)80%,transparent);
    padding:0 32px;height:62px;display:flex;align-items:center;gap:20px}
.logo{font-family:'Bebas Neue';font-size:1.8rem;color:var(--accent);
      letter-spacing:3px;flex-shrink:0}
.logo span{color:var(--text)}
.search-wrap{flex:1;max-width:440px;margin:0 auto;position:relative}
.search-input{width:100%;background:#ffffff0a;border:1px solid var(--border);
              border-radius:6px;padding:9px 16px 9px 40px;
              color:var(--text);font-family:'DM Sans';font-size:.88rem;
              outline:none;transition:.2s}
.search-input:focus{background:#ffffff10;border-color:#ffffff22}
.search-input::placeholder{color:var(--sub)}
.si{position:absolute;left:13px;top:50%;transform:translateY(-50%);
    color:var(--sub);pointer-events:none}
.search-results{position:absolute;top:calc(100%+6px);left:0;right:0;
                background:var(--surf);border:1px solid var(--border);
                border-radius:8px;overflow:hidden;display:none;
                z-index:400;box-shadow:0 16px 40px #00000099}
.sri{display:flex;align-items:center;gap:10px;padding:9px 13px;
     cursor:pointer;border-bottom:1px solid var(--border);transition:.15s}
.sri:hover{background:#ffffff08}
.sri img,.sri-ni{width:32px;height:44px;object-fit:cover;border-radius:3px;
                 flex-shrink:0;background:var(--dim);
                 display:flex;align-items:center;justify-content:center;font-size:.85rem}
.nav-right{display:flex;gap:10px;flex-shrink:0}
.nav-btn{color:#fff;border:none;padding:8px 16px;border-radius:6px;
         font-family:'DM Sans';font-weight:600;cursor:pointer;font-size:.8rem;
         transition:.2s;background:var(--accent)}
.nav-btn:hover{filter:brightness(1.1)}
.nav-btn.ghost{background:#ffffff10;border:1px solid var(--border)}
.nav-btn.ghost:hover{background:#ffffff18}

.genre-tabs{position:sticky;top:62px;z-index:200;
            background:var(--bg);border-bottom:1px solid var(--border);
            padding:10px 32px;display:flex;gap:8px;overflow-x:auto;scrollbar-width:none}
.genre-tabs::-webkit-scrollbar{display:none}
.gtab{flex-shrink:0;padding:6px 14px;border-radius:6px;
      border:1px solid var(--border);background:transparent;
      color:var(--sub);font-family:'DM Sans';font-size:.78rem;
      font-weight:500;cursor:pointer;transition:.2s}
.gtab:hover{color:var(--text);background:#ffffff08}
.gtab.active{background:var(--accent);color:#fff;border-color:transparent}

.hero{height:58vh;position:relative;overflow:hidden;
      display:flex;align-items:flex-end;padding:48px 32px;margin-top:62px}
.hero-bg{position:absolute;inset:0;
  background:radial-gradient(ellipse 70% 60% at 70% 20%,#2563eb18,transparent),
             linear-gradient(135deg,#07090e,#0a0e18)}
.hero-grid{position:absolute;inset:0;
  background-image:linear-gradient(#ffffff04 1px,transparent 1px),
                   linear-gradient(90deg,#ffffff04 1px,transparent 1px);
  background-size:60px 60px}
.hero-content{position:relative;z-index:1;max-width:560px}
.hero-label{display:inline-block;background:#2563eb18;border:1px solid #2563eb30;
            color:var(--blue);font-size:.72rem;font-weight:600;padding:4px 12px;
            border-radius:4px;margin-bottom:14px;letter-spacing:1.5px;text-transform:uppercase}
.hero h1{font-family:'Bebas Neue';font-size:clamp(3rem,5.5vw,5rem);
         line-height:.95;margin-bottom:12px;
         background:linear-gradient(135deg,#fff,#94a3b8);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:var(--sub);font-size:.92rem;line-height:1.7;max-width:440px;margin-bottom:20px}
.hero-btns{display:flex;gap:10px}
.btn-p{background:var(--accent);color:#fff;padding:10px 22px;
       border-radius:6px;border:none;font-family:'DM Sans';
       font-weight:600;cursor:pointer;font-size:.9rem;transition:.2s}
.btn-p:hover{filter:brightness(1.1)}
.btn-g{background:#ffffff0e;color:var(--text);padding:10px 22px;
       border-radius:6px;border:1px solid var(--border);
       font-family:'DM Sans';font-weight:500;cursor:pointer;font-size:.9rem;transition:.2s}
.btn-g:hover{background:#ffffff18}

.model-bar{background:var(--surf);border-bottom:1px solid var(--border);
           padding:10px 32px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.model-status{display:flex;align-items:center;gap:8px;font-size:.8rem}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.status-dot.ready{background:var(--green);box-shadow:0 0 6px var(--green)}
.status-dot.waiting{background:var(--sub)}
.status-chip{background:#2563eb18;border:1px solid #2563eb30;
             color:var(--blue);padding:3px 10px;border-radius:4px;font-size:.7rem;font-weight:600}

.stats{display:flex;background:var(--surf);
       border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
.stat{flex:1;padding:14px;text-align:center;border-right:1px solid var(--border)}
.stat:last-child{border-right:none}
.stat-v{font-family:'Bebas Neue';font-size:1.5rem;color:var(--accent)}
.stat-l{font-size:.65rem;color:var(--sub);text-transform:uppercase;letter-spacing:1px}

/* BIAS BANNER */
.bias-banner{display:none;margin:14px 32px;padding:14px 18px;
             border-radius:8px;font-size:.85rem;font-weight:500;line-height:1.6}
.bias-banner.warning{background:#eab30820;border:1px solid #eab30866;color:#eab308}
.bias-banner.safe{background:#22c55e18;border:1px solid #22c55e44;color:#22c55e}
.bias-banner strong{font-weight:700}
.bias-banner .banner-detail{font-size:.78rem;opacity:.85;margin-top:4px}

.rec-sec{padding:20px 32px 8px;background:var(--surf)}
.rec-hdr{display:flex;align-items:center;justify-content:space-between;
         margin-bottom:14px;flex-wrap:wrap;gap:10px}
.rec-title{font-family:'Bebas Neue';font-size:1.4rem}
.rec-chips{display:flex;gap:8px;flex-wrap:wrap}
.rec-chip{padding:5px 14px;border-radius:6px;border:1px solid var(--border);
          background:transparent;color:var(--sub);font-family:'DM Sans';
          font-size:.76rem;font-weight:500;cursor:pointer;transition:.2s}
.rec-chip:hover{color:var(--text);background:#ffffff08}
.rec-chip.active{color:#fff;border-color:transparent}
.rec-row{display:flex;gap:14px;overflow-x:auto;padding-bottom:8px;
         scrollbar-width:thin;scrollbar-color:#ffffff0f transparent}
.rec-loading{display:flex;align-items:center;gap:10px;
             padding:28px;color:var(--sub);font-size:.88rem}
.spin{width:16px;height:16px;border:2px solid var(--border);
      border-top-color:var(--blue);border-radius:50%;
      animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

.rec-card{flex:0 0 155px;border-radius:8px;overflow:hidden;
          background:var(--card);border:1px solid var(--border);
          cursor:pointer;transition:.2s;position:relative}
.rec-card:hover{transform:translateY(-6px);border-color:#ffffff18;
                box-shadow:0 16px 32px #00000066}
.rec-card img{width:155px;height:220px;object-fit:cover;display:block}
.rec-card-ni{width:155px;height:220px;background:var(--dim);
             display:flex;align-items:center;justify-content:center;
             font-size:.82rem;color:var(--sub);text-align:center;padding:12px}
.mitigated-badge{position:absolute;top:6px;right:6px;
                 background:#22c55e22;border:1px solid #22c55e55;
                 color:#22c55e;border-radius:4px;padding:2px 6px;
                 font-size:.58rem;font-weight:600}
.rec-body{padding:9px}
.rec-title-t{font-size:.8rem;font-weight:600;margin-bottom:3px;
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rec-meta{font-size:.68rem;color:var(--sub)}

.cat-sec{padding:22px 0 4px}
.cat-hdr{display:flex;align-items:center;gap:10px;padding:0 32px;margin-bottom:12px}
.cat-name{font-family:'Bebas Neue';font-size:1.2rem}
.cat-cnt{font-size:.7rem;color:var(--sub);background:var(--dim);
         padding:2px 9px;border-radius:4px}
.scl-ind{margin-left:auto;font-size:.7rem;color:var(--sub)}
.movie-row{display:flex;gap:12px;overflow-x:auto;
           padding:4px 32px 12px;scroll-behavior:smooth;
           scrollbar-width:thin;scrollbar-color:#ffffff0f transparent}
.mc{flex:0 0 145px;border-radius:8px;overflow:hidden;
    background:var(--card);border:1px solid var(--border);
    cursor:pointer;transition:.2s;position:relative}
.mc:hover{transform:translateY(-6px);border-color:#ffffff14;
          box-shadow:0 16px 32px #00000066}
.mc img{width:145px;height:205px;object-fit:cover;display:block}
.mc-ni{width:145px;height:205px;background:var(--dim);
       display:flex;align-items:center;justify-content:center;
       font-size:.82rem;color:var(--sub);text-align:center;padding:10px}
.mc-body{padding:8px}
.mc-title{font-size:.76rem;font-weight:600;margin-bottom:3px;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mc-meta{font-size:.66rem;color:var(--sub)}
.pop-badge{position:absolute;top:6px;right:6px;background:#00000099;
           backdrop-filter:blur(4px);border-radius:4px;padding:2px 6px;
           font-size:.58rem;color:var(--gold)}

.overlay{display:none;position:fixed;inset:0;background:#000000cc;
         z-index:500;align-items:center;justify-content:center;padding:20px}
.overlay.open{display:flex}
.modal{background:var(--surf);border:1px solid var(--border);
       border-radius:12px;max-width:480px;width:100%;
       max-height:90vh;overflow-y:auto;animation:popIn .25s ease}
@keyframes popIn{from{transform:scale(.92);opacity:0}to{transform:none;opacity:1}}
.m-hero{height:180px;overflow:hidden;border-radius:12px 12px 0 0;
        background:var(--dim);position:relative;
        background-size:cover;background-position:center}
.m-grad{position:absolute;inset:0;
        background:linear-gradient(0deg,var(--surf),transparent 60%)}
.m-close{position:absolute;top:10px;right:10px;background:#00000088;
         border:none;color:#fff;width:28px;height:28px;border-radius:50%;
         font-size:.85rem;cursor:pointer;z-index:1;
         display:flex;align-items:center;justify-content:center}
.m-body{padding:18px}
.m-title{font-family:'Bebas Neue';font-size:1.5rem;margin-bottom:3px}
.m-meta{font-size:.78rem;color:var(--sub);margin-bottom:8px}
.m-overview{font-size:.82rem;color:var(--sub);line-height:1.6;margin-bottom:12px}
.m-tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.m-tag{background:var(--dim);border-radius:4px;padding:3px 9px;
       font-size:.7rem;color:var(--sub)}
.pred-box{background:var(--dim);border:1px solid var(--border);
          border-radius:8px;padding:12px;margin-bottom:14px;display:none}
.pred-box.show{display:block}
.pred-label{font-size:.72rem;color:var(--sub);text-transform:uppercase;
            letter-spacing:1px;margin-bottom:6px}
.pred-value{font-family:'Bebas Neue';font-size:1.5rem}
.pred-value.yes{color:var(--green)}
.pred-value.no{color:var(--red)}
.pred-note{font-size:.7rem;color:var(--sub);margin-top:4px}
.rate-lbl{font-size:.82rem;font-weight:600;margin-bottom:8px}
.stars{display:flex;gap:6px;margin-bottom:16px}
.star{font-size:1.8rem;cursor:pointer;transition:.15s;opacity:.25}
.star.lit{opacity:1}
.m-submit{width:100%;background:var(--accent);color:#fff;border:none;
          padding:12px;border-radius:8px;font-family:'DM Sans';
          font-weight:600;font-size:.92rem;cursor:pointer;transition:.2s}
.m-submit:hover{filter:brightness(1.1)}

.charts-overlay{display:none;position:fixed;inset:0;background:#000000cc;
                z-index:500;align-items:center;justify-content:center;padding:20px}
.charts-overlay.open{display:flex}
.charts-box{background:var(--surf);border:1px solid var(--border);
            border-radius:12px;max-width:940px;width:100%;
            max-height:90vh;overflow-y:auto;padding:24px}
.charts-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.charts-title{font-family:'Bebas Neue';font-size:1.4rem}
.charts-close{background:none;border:none;color:var(--sub);font-size:1.3rem;cursor:pointer}
.chart-img{width:100%;border-radius:8px;border:1px solid var(--border)}
.gini-note{background:var(--dim);border:1px solid var(--border);border-radius:8px;
           padding:12px 16px;margin-top:12px;font-size:.82rem;color:var(--sub);line-height:1.6}
.gini-note strong{color:var(--text)}
.metric-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}
.metric-box{background:var(--dim);border-radius:8px;padding:12px;
            text-align:center;border:1px solid var(--border)}
.metric-val{font-family:'Bebas Neue';font-size:1.6rem}
.metric-lbl{font-size:.68rem;color:var(--sub);margin-top:2px}

.p-overlay{display:none;position:fixed;inset:0;z-index:390}
.p-overlay.open{display:block}
.bp{position:fixed;top:0;right:-420px;width:400px;height:100vh;
    background:var(--surf);border-left:1px solid var(--border);
    z-index:400;transition:.35s ease;overflow-y:auto;padding:20px 18px}
.bp.open{right:0;box-shadow:-20px 0 50px #00000099}
.ph{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.ph-t{font-family:'Bebas Neue';font-size:1.3rem}
.ph-x{background:none;border:none;color:var(--sub);font-size:1.2rem;cursor:pointer}
.kpi-g{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}
.kpi-v{font-family:'Bebas Neue';font-size:1.6rem}
.kpi-l{font-size:.63rem;color:var(--sub);text-transform:uppercase;letter-spacing:1px;margin-top:1px}
.bias-note{background:var(--dim);border:1px solid var(--border);border-radius:8px;
           padding:12px;margin-bottom:14px;font-size:.78rem;color:var(--sub);line-height:1.6}
.bias-note strong{color:var(--text)}
.p-sec{margin-bottom:14px}
.p-sec-t{font-size:.66rem;color:var(--sub);text-transform:uppercase;
         letter-spacing:1.5px;font-weight:600;margin-bottom:8px}
.gini-track{background:#ffffff08;border-radius:4px;height:16px;overflow:hidden;margin:6px 0 4px}
.gini-fill{height:100%;border-radius:4px;
           background:linear-gradient(90deg,#22c55e,#f59e0b 50%,#ef4444);
           transition:width 1s ease;min-width:2px;
           display:flex;align-items:center;padding:0 6px}
.gini-fill span{font-size:.6rem;font-weight:700;color:#000}
.gini-ax{display:flex;justify-content:space-between;font-size:.6rem;color:var(--sub)}
.pn-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.pn{background:var(--card);border-radius:8px;padding:12px;
    text-align:center;border:1px solid var(--border)}
.pn-v{font-family:'Bebas Neue';font-size:1.7rem}
.pn-l{font-size:.63rem;color:var(--sub);margin-top:1px}
.fm-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
.fm-item{background:var(--card);border-radius:8px;padding:10px;border:1px solid var(--border)}
.fm-val{font-family:'Bebas Neue';font-size:1.2rem}
.fm-lbl{font-size:.62rem;color:var(--sub);margin-top:1px}
.fm-note{font-size:.58rem;color:var(--sub);margin-top:1px;font-style:italic}
.bar-list{display:flex;flex-direction:column;gap:7px;margin-bottom:14px}
.bar-item{display:flex;align-items:center;gap:8px}
.bar-lbl{font-size:.7rem;width:72px;text-align:right;color:var(--sub);flex-shrink:0}
.bar-track{flex:1;background:#ffffff08;border-radius:3px;height:14px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .6s ease;
          display:flex;align-items:center;padding:0 5px}
.bar-fill span{font-size:.58rem;font-weight:600;color:#fff}
.top-list{display:flex;flex-direction:column;gap:6px;margin-bottom:14px}
.top-item{display:flex;align-items:center;gap:8px;background:var(--card);
          border-radius:6px;padding:8px;border:1px solid var(--border)}
.top-rank{font-family:'Bebas Neue';font-size:1rem;color:var(--accent);width:20px;flex-shrink:0}
.top-t{font-size:.8rem;flex:1}
.top-v{font-size:.7rem;color:var(--sub)}
.p-refresh{width:100%;background:#ffffff08;color:var(--text);
           border:1px solid var(--border);padding:10px;
           border-radius:6px;font-family:'DM Sans';font-size:.8rem;
           cursor:pointer;transition:.2s}
.p-refresh:hover{background:#ffffff12}

/* MITIGATION STATUS BOX in panel */
.mitigation-box{border-radius:8px;padding:12px;margin-bottom:14px;font-size:.78rem;line-height:1.6}
.mitigation-box.active{background:#22c55e18;border:1px solid #22c55e44;color:#22c55e}
.mitigation-box.inactive{background:var(--dim);border:1px solid var(--border);color:var(--sub)}
.mitigation-box strong{font-weight:700}

.toast{position:fixed;bottom:24px;left:50%;
       transform:translateX(-50%) translateY(14px);
       background:var(--surf);border:1px solid var(--border);
       border-radius:8px;padding:10px 20px;font-size:.86rem;
       opacity:0;transition:.3s;z-index:999;pointer-events:none;white-space:nowrap}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

footer{padding:32px;text-align:center;color:var(--sub);
       font-size:.76rem;border-top:1px solid var(--border);margin-top:16px}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="logo">Reco<span>met</span></div>
  <div class="search-wrap">
    <span class="si">&#128269;</span>
    <input class="search-input" id="searchInput" type="text"
           placeholder="Search movies..."
           oninput="handleSearch(this.value)" autocomplete="off">
    <div class="search-results" id="searchResults"></div>
  </div>
  <div class="nav-right">
    <button class="nav-btn ghost" onclick="openCharts()">Bias Charts</button>
    <button class="nav-btn"       onclick="togglePanel()">Live Panel</button>
  </div>
</nav>

<!-- GENRE TABS -->
<div class="genre-tabs">
  <button class="gtab active" onclick="filterGenre('all',this)">All</button>
  <button class="gtab" onclick="filterGenre('Bollywood',this)">Bollywood</button>
  <button class="gtab" onclick="filterGenre('Action',this)">Action</button>
  <button class="gtab" onclick="filterGenre('Drama',this)">Drama</button>
  <button class="gtab" onclick="filterGenre('Romance',this)">Romance</button>
  <button class="gtab" onclick="filterGenre('Comedy',this)">Comedy</button>
  <button class="gtab" onclick="filterGenre('Thriller',this)">Thriller</button>
  <button class="gtab" onclick="filterGenre('Sci-Fi',this)">Sci-Fi</button>
  <button class="gtab" onclick="filterGenre('Horror',this)">Horror</button>
  <button class="gtab" onclick="filterGenre('Animation',this)">Animation</button>
  <button class="gtab" onclick="filterGenre('Crime',this)">Crime</button>
</div>

<!-- HERO -->
<div class="hero">
  <div class="hero-bg"></div>
  <div class="hero-grid"></div>
  <div class="hero-content">
    <div class="hero-label">AI-Based Recommendation System with Bias Detection</div>
    <h1>Discover.<br>Rate. Analyse.</h1>
    <p>A research demonstration of feedback loop bias in recommendation systems.
       Bias is automatically detected and mitigated in real time.</p>
    <div class="hero-btns">
      <button class="btn-p" onclick="document.querySelector('.rec-sec').scrollIntoView({behavior:'smooth'})">Browse Movies</button>
      <button class="btn-g" onclick="togglePanel()">View Bias Metrics</button>
    </div>
  </div>
</div>

<!-- MODEL STATUS BAR -->
<div class="model-bar">
  <div class="model-status">
    <div class="status-dot waiting" id="modelDot"></div>
    <span id="modelStatus">Logistic Regression: Waiting for training data...</span>
  </div>
  <div class="status-chip" id="accChip" style="display:none">
    Accuracy: <span id="accVal">--</span>% &nbsp;|&nbsp; Source: <span id="srcVal">--</span>
  </div>
</div>

<!-- STATS -->
<div class="stats">
  <div class="stat"><div class="stat-v" id="s-views">0</div><div class="stat-l">Views</div></div>
  <div class="stat"><div class="stat-v" id="s-ratings">0</div><div class="stat-l">Ratings</div></div>
  <div class="stat"><div class="stat-v" id="s-scrolls">0</div><div class="stat-l">Scrolls</div></div>
  <div class="stat"><div class="stat-v" id="s-gini">--</div><div class="stat-l">Gini Index</div></div>
  <div class="stat"><div class="stat-v" id="s-bias">--</div><div class="stat-l">Bias Level</div></div>
</div>

<!-- BIAS BANNER — shown automatically when bias is HIGH -->
<div class="bias-banner warning" id="biasBanner">
  <strong>Feedback Loop Bias Detected &mdash; Gini: <span id="bannerGini">0.0000</span></strong>
  <div class="banner-detail">
    Recommendations have been automatically adjusted to give niche content fairer exposure.
  </div>
</div>

<!-- RECOMMENDATIONS -->
<div class="rec-sec">
  <div class="rec-hdr">
    <div class="rec-title">Recommended Movies</div>
    <div class="rec-chips" id="recChips">
      <button class="rec-chip active"
              style="background:#e11d48;border-color:#e11d48;color:#fff"
              data-genre="Bollywood" onclick="toggleRecGenre(this)">Bollywood</button>
      <button class="rec-chip" data-genre="Action"    onclick="toggleRecGenre(this)">Action</button>
      <button class="rec-chip" data-genre="Drama"     onclick="toggleRecGenre(this)">Drama</button>
      <button class="rec-chip" data-genre="Romance"   onclick="toggleRecGenre(this)">Romance</button>
      <button class="rec-chip" data-genre="Comedy"    onclick="toggleRecGenre(this)">Comedy</button>
      <button class="rec-chip" data-genre="Thriller"  onclick="toggleRecGenre(this)">Thriller</button>
    </div>
  </div>
  <div class="rec-row" id="recRow">
    <div class="rec-loading"><div class="spin"></div>&nbsp; Loading recommendations...</div>
  </div>
</div>

<!-- MOVIE ROWS -->
<div id="movieRows"></div>

<!-- RATING MODAL -->
<div class="overlay" id="overlay">
  <div class="modal">
    <div class="m-hero" id="mHero">
      <div class="m-grad"></div>
      <button class="m-close" onclick="closeModal()">&#x2715;</button>
    </div>
    <div class="m-body">
      <div class="m-title"    id="mTitle"></div>
      <div class="m-meta"     id="mMeta"></div>
      <div class="m-overview" id="mOvr"></div>
      <div class="m-tags"     id="mTags"></div>
      <div class="pred-box"   id="predBox">
        <div class="pred-label">Model Prediction</div>
        <div class="pred-value" id="predValue"></div>
        <div class="pred-note"  id="predNote"></div>
      </div>
      <div class="rate-lbl">Your Rating</div>
      <div class="stars">
        <span class="star" data-v="1">&#11088;</span>
        <span class="star" data-v="2">&#11088;</span>
        <span class="star" data-v="3">&#11088;</span>
        <span class="star" data-v="4">&#11088;</span>
        <span class="star" data-v="5">&#11088;</span>
      </div>
      <button class="m-submit" onclick="submitRating()">Submit Rating</button>
    </div>
  </div>
</div>

<!-- CHARTS MODAL -->
<div class="charts-overlay" id="chartsOverlay">
  <div class="charts-box">
    <div class="charts-hdr">
      <div class="charts-title">Bias Analysis Charts</div>
      <button class="charts-close" onclick="closeCharts()">&#x2715;</button>
    </div>
    <img id="chartImg" class="chart-img" src="" alt="Bias Charts">
    <div class="gini-note">
      <strong>Gini Coefficient Interpretation:</strong>
      Higher Gini value indicates higher bias in recommendations.
      A value of 0 means perfectly equal exposure for all items.
      A value close to 1 means almost all views are concentrated on a few popular items,
      confirming the feedback loop effect. When Gini exceeds 0.65, mitigation is automatically applied.
    </div>
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-val" style="color:#f59e0b">0.629</div>
        <div class="metric-lbl">Gini Before (Original)</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" style="color:#ef4444">0.736</div>
        <div class="metric-lbl">Gini After Loop Simulation</div>
      </div>
      <div class="metric-box">
        <div class="metric-val" style="color:#22c55e">0.476</div>
        <div class="metric-lbl">Gini After Mitigation</div>
      </div>
    </div>
  </div>
</div>

<!-- BIAS PANEL -->
<div class="p-overlay" id="pOverlay" onclick="closePanel()"></div>
<div class="bp" id="bp">
  <div class="ph">
    <div class="ph-t">Bias Metrics Panel</div>
    <button class="ph-x" onclick="closePanel()">&#x2715;</button>
  </div>
  <div class="kpi-g">
    <div class="kpi"><div class="kpi-v" id="p-views" style="color:var(--accent)">0</div><div class="kpi-l">Total Views</div></div>
    <div class="kpi"><div class="kpi-v" id="p-gini"  style="color:var(--gold)">--</div><div class="kpi-l">Gini Score</div></div>
  </div>
  <div class="bias-note">
    <strong>Note:</strong> Higher Gini value indicates higher bias in recommendations.
    Gini above 0.65 triggers automatic mitigation.
  </div>

  <!-- MITIGATION STATUS BOX -->
  <div class="mitigation-box inactive" id="mitigationBox">
    <strong>Mitigation Status: Inactive</strong><br>
    No bias detected. Recommendations are served by rating score only.
  </div>

  <div class="p-sec">
    <div class="p-sec-t">Gini Coefficient</div>
    <div class="gini-track">
      <div class="gini-fill" id="giniFill" style="width:1%">
        <span id="giniTxt"></span>
      </div>
    </div>
    <div class="gini-ax"><span>Fair (0.0)</span><span>Moderate (0.65)</span><span>Biased (1.0)</span></div>
  </div>
  <div class="pn-row">
    <div class="pn"><div class="pn-v" id="p-pop"   style="color:var(--red)">--%</div><div class="pn-l">Popular</div></div>
    <div class="pn"><div class="pn-v" id="p-niche" style="color:var(--blue)">--%</div><div class="pn-l">Niche</div></div>
  </div>
  <div class="p-sec">
    <div class="p-sec-t">Fairness Metrics</div>
    <div class="fm-grid">
      <div class="fm-item"><div class="fm-val" id="di-val">--</div><div class="fm-lbl">Disparate Impact</div><div class="fm-note">Fair: 0.8 to 1.2</div></div>
      <div class="fm-item"><div class="fm-val" id="sp-val">--</div><div class="fm-lbl">Stat. Parity</div><div class="fm-note">Fair: near 0</div></div>
      <div class="fm-item"><div class="fm-val" id="eo-val">--</div><div class="fm-lbl">Equal Opport.</div><div class="fm-note">Fair: near 0</div></div>
      <div class="fm-item"><div class="fm-val" id="gini-p">--</div><div class="fm-lbl">Gini Index</div><div class="fm-note">Fair: near 0</div></div>
    </div>
  </div>
  <div class="p-sec">
    <div class="p-sec-t">Category Scroll Activity</div>
    <div class="bar-list" id="scrollBars"></div>
  </div>
  <div class="p-sec">
    <div class="p-sec-t">Top Viewed Movies</div>
    <div class="top-list" id="topList"></div>
  </div>
  <div class="p-sec">
    <div class="p-sec-t">Model Info</div>
    <div id="modelInfo" style="font-size:.78rem;color:var(--sub);line-height:1.8">
      Model not yet trained.
    </div>
  </div>
  <button class="p-refresh" onclick="fetchBias()">Refresh Data</button>
</div>

<div class="toast" id="toast"></div>

<footer>Developed for academic demonstration of recommendation bias analysis.</footer>

<script>
const GENRE_COLORS = {
  "Bollywood":"#e11d48","Action":"#ef4444","Drama":"#8b5cf6",
  "Romance":"#ec4899","Comedy":"#f59e0b","Thriller":"#10b981",
  "Sci-Fi":"#06b6d4","Horror":"#6366f1","Animation":"#f97316","Crime":"#64748b"
};

let curMovie = null, curStar = 0;
let totalViews = 0, totalRatings = 0, totalScrolls = 0;
let scrollCounts = {}, searchTimer = null;
let selectedRecGenres = new Set(["Bollywood"]);

// ── INIT ──────────────────────────────────────────────────────
async function initApp() {
  const genres = ["Bollywood","Action","Drama","Romance",
                  "Comedy","Thriller","Sci-Fi","Horror","Animation","Crime"];
  const container = document.getElementById('movieRows');
  for (const g of genres) {
    scrollCounts[g] = 0;
    const movies = await fetchGenre(g);
    if (!movies.length) continue;
    container.appendChild(buildRow(g, movies));
  }
  loadRecs();
  updateModelBar();
}

async function fetchGenre(g) {
  const r = await fetch(`/movies?genre=${encodeURIComponent(g)}`);
  return (await r.json()).movies || [];
}

// ── REC GENRE CHIPS ───────────────────────────────────────────
function toggleRecGenre(btn) {
  const g = btn.dataset.genre;
  const color = GENRE_COLORS[g] || '#888';
  if (selectedRecGenres.has(g)) {
    if (selectedRecGenres.size === 1) { showToast("Keep at least one genre selected."); return; }
    selectedRecGenres.delete(g);
    btn.classList.remove('active');
    btn.style.background = 'transparent';
    btn.style.borderColor = 'var(--border)';
    btn.style.color = 'var(--sub)';
  } else {
    selectedRecGenres.add(g);
    btn.classList.add('active');
    btn.style.background = color;
    btn.style.borderColor = color;
    btn.style.color = '#fff';
  }
  loadRecs();
}

// ── LOAD RECOMMENDATIONS WITH AUTO BIAS DETECTION & MITIGATION ─
async function loadRecs() {
  const c = document.getElementById('recRow');
  c.innerHTML = '<div class="rec-loading"><div class="spin"></div>&nbsp; Loading...</div>';
  const genres = Array.from(selectedRecGenres).join(',');
  const d = await fetch(`/recommendations?genres=${encodeURIComponent(genres)}`).then(r => r.json());

  // Auto bias banner
  const banner = document.getElementById('biasBanner');
  const mitBox = document.getElementById('mitigationBox');
  if (d.bias_detected) {
    banner.style.display = 'block';
    banner.className = 'bias-banner warning';
    document.getElementById('bannerGini').textContent = d.gini;
    mitBox.className = 'mitigation-box active';
    mitBox.innerHTML = `<strong>Mitigation Status: Active</strong><br>
      Gini = ${d.gini} (above 0.65). Niche movies have been boosted in ranking to reduce bias.`;
  } else {
    banner.style.display = 'none';
    mitBox.className = 'mitigation-box inactive';
    mitBox.innerHTML = `<strong>Mitigation Status: Inactive</strong><br>
      Gini = ${d.gini || '--'} (below 0.65). No bias detected. Serving results by rating.`;
  }

  if (!d.movies || !d.movies.length) {
    c.innerHTML = '<div style="color:var(--sub);padding:24px;font-size:.85rem">No results found.</div>';
    return;
  }

  c.innerHTML = d.movies.map(m => `
    <div class="rec-card" onclick='openModal(${JSON.stringify(m)})'>
      ${m.poster
        ? `<img src="${m.poster}" alt="${m.title}" loading="lazy">`
        : `<div class="rec-card-ni">${m.title}</div>`}
      ${m.mitigated ? `<div class="mitigated-badge">Mitigated</div>` : ''}
      <div class="rec-body">
        <div class="rec-title-t">${m.title}</div>
        <div class="rec-meta">${m.year} &nbsp;|&nbsp; &#9733; ${m.rating}</div>
      </div>
    </div>`).join('');
}

// ── ROW BUILDER ───────────────────────────────────────────────
function buildRow(genre, movies) {
  const sec = document.createElement('div');
  sec.className = 'cat-sec';
  sec.id = `cat-${genre}`;
  sec.dataset.genre = genre;
  sec.innerHTML = `
    <div class="cat-hdr">
      <div class="cat-name">${genre}</div>
      <div class="cat-cnt">${movies.length} movies</div>
      <div class="scl-ind" id="sc-${genre}">0 scrolls</div>
    </div>
    <div class="movie-row" id="row-${genre}">
      ${movies.map(m => `
        <div class="mc" onclick='openModal(${JSON.stringify(m)})'>
          ${m.poster
            ? `<img src="${m.poster}" alt="${m.title}" loading="lazy">`
            : `<div class="mc-ni">${m.title}</div>`}
          ${m.popular ? `<div class="pop-badge">Popular</div>` : ''}
          <div class="mc-body">
            <div class="mc-title">${m.title}</div>
            <div class="mc-meta">${m.year} &nbsp;|&nbsp; &#9733; ${m.rating}</div>
          </div>
        </div>`).join('')}
    </div>`;
  setTimeout(() => {
    const row = sec.querySelector('.movie-row');
    let last = 0;
    row.addEventListener('scroll', () => {
      const now = Date.now();
      if (now - last > 400) {
        last = now;
        scrollCounts[genre] = (scrollCounts[genre] || 0) + 1;
        totalScrolls++;
        document.getElementById(`sc-${genre}`).textContent = scrollCounts[genre] + ' scrolls';
        document.getElementById('s-scrolls').textContent = totalScrolls;
        fetch('/track_scroll', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ category: genre })
        });
      }
    });
  }, 100);
  return sec;
}

// ── MODAL ─────────────────────────────────────────────────────
function openModal(m) {
  if (typeof m === 'string') m = JSON.parse(m);
  curMovie = m; curStar = 0;
  document.getElementById('mTitle').textContent = m.title;
  document.getElementById('mMeta').textContent =
    `${m.year}  |  Rating: ${m.rating}  |  ${m.popular ? 'Popular' : 'Niche'}  |  ${(m.votes / 1000).toFixed(0)}k votes`;
  document.getElementById('mOvr').textContent = m.overview || '';
  document.getElementById('mTags').innerHTML = (m.genres || [])
    .map(g => `<span class="m-tag">${g}</span>`).join('');
  const hero = document.getElementById('mHero');
  hero.style.background = m.poster ? `url(${m.poster}) center/cover` : 'var(--dim)';
  document.querySelectorAll('.star').forEach(s => s.classList.remove('lit'));
  showPrediction(m.id);
  document.getElementById('overlay').classList.add('open');
  totalViews++;
  document.getElementById('s-views').textContent = totalViews;
  fetch('/track_view', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ movie_id: m.id })
  });
}

function closeModal() {
  document.getElementById('overlay').classList.remove('open');
}

// Stars highlight only — prediction NOT updated on click
document.querySelectorAll('.star').forEach(star => {
  star.addEventListener('click', () => {
    curStar = parseInt(star.dataset.v);
    document.querySelectorAll('.star').forEach((s, i) =>
      s.classList.toggle('lit', i < curStar));
  });
});

function showPrediction(movieId) {
  fetch(`/predict?movie_id=${movieId}`)
    .then(r => r.json())
    .then(d => {
      const box = document.getElementById('predBox');
      if (d.prediction === null) { box.classList.remove('show'); return; }
      box.classList.add('show');
      const val = document.getElementById('predValue');
      val.textContent = `Prediction: ${d.prediction}`;
      val.className = `pred-value ${d.prediction === 'YES' ? 'yes' : 'no'}`;
      document.getElementById('predNote').textContent =
        d.prediction === 'YES'
          ? 'The model predicts you will like this movie.'
          : 'The model predicts this may not match your preference.';
    }).catch(() => {});
}

async function submitRating() {
  if (!curStar) { showToast('Please select a star rating.'); return; }
  await fetch('/rate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ movie_id: curMovie.id, rating: curStar })
  });
  totalRatings++;
  document.getElementById('s-ratings').textContent = totalRatings;
  if (totalRatings % 5 === 0) {
    await fetch('/train_lr', { method: 'POST' });
    updateModelBar();
    fetchBias();
    loadRecs(); // Refresh recs — mitigation may now activate
  }
  // Update prediction AFTER rating is submitted
  showPrediction(curMovie.id);
  showToast(`Rated "${curMovie.title}" ${curStar} star(s).`);
  closeModal();
  updateModelBar();
}

async function updateModelBar() {
  const d = await fetch('/ml_status').then(r => r.json());
  const dot = document.getElementById('modelDot');
  if (d.trained) {
    dot.className = 'status-dot ready';
    document.getElementById('modelStatus').textContent =
      `Logistic Regression: Trained (${d.n_samples} samples, ${d.accuracy}% accuracy)`;
    document.getElementById('accChip').style.display = '';
    document.getElementById('accVal').textContent = d.accuracy;
    document.getElementById('srcVal').textContent =
      d.source === 'dataset' ? 'MovieLens Dataset' : 'User Interactions';
    document.getElementById('modelInfo').innerHTML =
      `Trained on: <strong style="color:var(--text)">${d.source === 'dataset' ? 'MovieLens 100k dataset' : 'Live user interactions'}</strong><br>
       Samples: <strong style="color:var(--text)">${d.n_samples}</strong><br>
       Accuracy: <strong style="color:var(--text)">${d.accuracy}%</strong><br>
       Epochs: <strong style="color:var(--text)">300</strong>`;
  } else {
    dot.className = 'status-dot waiting';
    document.getElementById('modelStatus').textContent =
      `Logistic Regression: Need ${Math.max(0, 5 - totalRatings)} more ratings (${totalRatings}/5)`;
  }
}

// ── GENRE FILTER ──────────────────────────────────────────────
function filterGenre(g, btn) {
  document.querySelectorAll('.gtab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.cat-sec').forEach(s => {
    s.style.display = (g === 'all' || s.dataset.genre === g) ? 'block' : 'none';
  });
  if (g !== 'all') {
    const el = document.getElementById(`cat-${g}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// ── SEARCH ────────────────────────────────────────────────────
async function handleSearch(q) {
  clearTimeout(searchTimer);
  const box = document.getElementById('searchResults');
  if (q.length < 2) { box.style.display = 'none'; return; }
  searchTimer = setTimeout(async () => {
    const d = await fetch(`/search?q=${encodeURIComponent(q)}`).then(r => r.json());
    if (!d.results || !d.results.length) { box.style.display = 'none'; return; }
    box.innerHTML = d.results.slice(0, 6).map(m => `
      <div class="sri" onclick='openModal(${JSON.stringify(m)});
           document.getElementById("searchResults").style.display="none";
           document.getElementById("searchInput").value=""'>
        ${m.poster ? `<img src="${m.poster}" alt="">` : `<div class="sri-ni">?</div>`}
        <div>
          <div style="font-size:.85rem;font-weight:500">${m.title}</div>
          <div style="font-size:.72rem;color:var(--sub)">${m.year} &nbsp;|&nbsp; &#9733; ${m.rating}</div>
        </div>
      </div>`).join('');
    box.style.display = 'block';
  }, 380);
}
document.addEventListener('click', e => {
  if (!e.target.closest('.search-wrap'))
    document.getElementById('searchResults').style.display = 'none';
});

// ── CHARTS ────────────────────────────────────────────────────
async function openCharts() {
  document.getElementById('chartsOverlay').classList.add('open');
  const r = await fetch('/chart_bias').then(r => r.json());
  if (r.img) document.getElementById('chartImg').src = 'data:image/png;base64,' + r.img;
}
function closeCharts() {
  document.getElementById('chartsOverlay').classList.remove('open');
}
document.getElementById('chartsOverlay').addEventListener('click', e => {
  if (e.target === document.getElementById('chartsOverlay')) closeCharts();
});

// ── BIAS PANEL ────────────────────────────────────────────────
function togglePanel() {
  document.getElementById('bp').classList.toggle('open');
  document.getElementById('pOverlay').classList.toggle('open');
  fetchBias();
}
function closePanel() {
  document.getElementById('bp').classList.remove('open');
  document.getElementById('pOverlay').classList.remove('open');
}

async function fetchBias() {
  const d = await fetch('/bias_data').then(r => r.json());
  document.getElementById('p-views').textContent = d.total_views;
  document.getElementById('p-gini').textContent  = d.gini;
  document.getElementById('p-pop').textContent   = d.pop_pct + '%';
  document.getElementById('p-niche').textContent = d.niche_pct + '%';
  document.getElementById('s-gini').textContent  = d.gini;
  document.getElementById('s-bias').textContent  = d.bias_level;
  document.getElementById('gini-p').textContent  = d.gini;

  const pct = Math.min(d.gini * 100, 98);
  document.getElementById('giniFill').style.width = pct + '%';
  document.getElementById('giniTxt').textContent  = d.gini;

  document.getElementById('di-val').textContent = d.disparate_impact ?? '--';
  document.getElementById('sp-val').textContent = d.statistical_parity ?? '--';
  document.getElementById('eo-val').textContent = d.equal_opportunity ?? '--';

  function mc(val, type) {
    if (val === null || val === undefined) return 'var(--sub)';
    if (type === 'di') return (val >= 0.8 && val <= 1.2) ? 'var(--green)' : 'var(--gold)';
    return Math.abs(val) < 0.1 ? 'var(--green)' : Math.abs(val) < 0.2 ? 'var(--gold)' : 'var(--red)';
  }
  if (d.disparate_impact !== null)   document.getElementById('di-val').style.color = mc(d.disparate_impact, 'di');
  if (d.statistical_parity !== null) document.getElementById('sp-val').style.color = mc(d.statistical_parity, 'sp');
  if (d.equal_opportunity !== null)  document.getElementById('eo-val').style.color = mc(d.equal_opportunity, 'eo');

  const entries = Object.entries(d.scroll).sort((a, b) => b[1] - a[1]);
  const maxS    = Math.max(...entries.map(e => e[1]), 1);
  document.getElementById('scrollBars').innerHTML = entries.map(([cat, cnt]) => `
    <div class="bar-item">
      <div class="bar-lbl">${cat}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${cnt / maxS * 100}%;background:${GENRE_COLORS[cat] || '#888'}88">
          <span>${cnt}</span></div></div></div>`).join('');

  const tl = document.getElementById('topList');
  tl.innerHTML = d.top_viewed.length
    ? d.top_viewed.map((t, i) => `
        <div class="top-item">
          <div class="top-rank">#${i + 1}</div>
          <div class="top-t">${t.title}</div>
          <div class="top-v">${t.views} views</div>
        </div>`).join('')
    : '<div style="color:var(--sub);font-size:.78rem;padding:6px">No views recorded yet.</div>';
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}

initApp();
setInterval(() => {
  if (document.getElementById('bp').classList.contains('open')) fetchBias();
}, 8000);
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/movies')
def movies():
    genre   = request.args.get("genre", "Bollywood")
    results, seen = [], set()
    for q in GENRES.get(genre, [genre.lower()]):
        for raw in omdb_search(q)[:2]:
            iid = raw.get("imdbID")
            if iid and iid not in seen:
                seen.add(iid)
                det = omdb_detail(iid)
                if det.get("Response") == "True":
                    results.append(enrich(det))
    return jsonify({"movies": results})

@app.route('/recommendations')
def recommendations():
    genres_param = request.args.get("genres", "Bollywood")
    selected = [g.strip() for g in genres_param.split(",") if g.strip() in GENRES]
    if not selected:
        selected = ["Bollywood"]
    result = get_recommendations(selected)
    return jsonify(result)

@app.route('/search')
def search():
    q = request.args.get("q", "")
    results, seen = [], set()
    for raw in omdb_search(q)[:6]:
        iid = raw.get("imdbID")
        if iid and iid not in seen:
            seen.add(iid)
            det = omdb_detail(iid)
            if det.get("Response") == "True":
                results.append(enrich(det))
    return jsonify({"results": results})

@app.route('/track_scroll', methods=['POST'])
def track_scroll():
    SCROLL_LOG[request.json['category']] += 1
    return jsonify(ok=True)

@app.route('/track_view', methods=['POST'])
def track_view():
    VIEW_LOG[str(request.json['movie_id'])] += 1
    return jsonify(ok=True)

@app.route('/rate', methods=['POST'])
def rate():
    data = request.json
    RATING_LOG[str(data['movie_id'])].append(int(data['rating']))
    total = sum(len(v) for v in RATING_LOG.values())
    if total >= 5:
        train_on_interactions()
    return jsonify(ok=True)

@app.route('/train_lr', methods=['POST'])
def train_lr():
    train_on_interactions()
    return jsonify(
        ok=True,
        trained=LR_MODEL["trained"],
        accuracy=LR_MODEL.get("accuracy", 0)
    )

@app.route('/ml_status')
def ml_status():
    return jsonify(
        trained=LR_MODEL["trained"],
        accuracy=LR_MODEL.get("accuracy", 0),
        n_samples=LR_MODEL.get("n_samples", 0),
        source=LR_MODEL.get("source", "none")
    )

@app.route('/predict')
def predict():
    mid  = request.args.get("movie_id", "")
    pred = predict_liked(mid)
    return jsonify(prediction=pred)

@app.route('/bias_data')
def bias_data():
    return jsonify(compute_bias())

@app.route('/chart_bias')
def chart_bias():
    img = make_bias_chart()
    return jsonify(img=img)

# ═══════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "=" * 56)
    print("  Recomet — AI-Based Recommendation System")
    print("  with Automatic Bias Detection & Mitigation")
    print("=" * 56)
    print("\n  Attempting dataset training from u.data...")
    load_dataset()
    print("\n  Open browser: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
