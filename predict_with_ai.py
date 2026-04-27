"""Mood prediction with AI fallback.

URL or 'artist - title' input. Catalog hit -> stored prediction. Catalog miss
-> Claude estimates audio features -> XGBoost -> prediction.
Either way, Claude generates a mood description and rationale.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 predict_with_ai.py "https://open.spotify.com/track/23PvWFdi76vER4p1e2Xroj"
  python3 predict_with_ai.py "Take Me Out by Franz Ferdinand"
  python3 predict_with_ai.py "Espresso by Sabrina Carpenter"
"""
import os, re, sys, json, html, urllib.request
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.neighbors import NearestNeighbors

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(OUT_DIR, "catalog.csv")
MODELS_DIR   = os.path.join(OUT_DIR, "models")
CLAUDE_MODEL = "claude-sonnet-4-5"
USE_MOCK_AI  = not bool(os.environ.get("ANTHROPIC_API_KEY"))

_catalog = None; _nn_index = None
_models = None; _genre_enc = None; _artist_enc = None; _feature_cols = None

def load_catalog():
    global _catalog, _nn_index
    if _catalog is not None: return _catalog
    _catalog = pd.read_csv(CATALOG_PATH).reset_index(drop=True)
    _nn_index = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        _catalog[["pred_valence", "pred_energy"]].values)
    return _catalog

def load_models():
    global _models, _genre_enc, _artist_enc, _feature_cols
    if _models is not None: return
    mv = XGBRegressor(); mv.load_model(f"{MODELS_DIR}/xgb_valence.json")
    me = XGBRegressor(); me.load_model(f"{MODELS_DIR}/xgb_energy.json")
    _models = {"valence": mv, "energy": me}
    _genre_enc = pd.read_csv(f"{MODELS_DIR}/genre_encoder.csv")
    with open(f"{MODELS_DIR}/artist_encoder.json") as f:
        _artist_enc = json.load(f)
    with open(f"{MODELS_DIR}/feature_columns.json") as f:
        _feature_cols = json.load(f)

def extract_track_id(s):
    m = re.search(r"track[/:]([A-Za-z0-9]{22})", s)
    if m: return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9]{22}", s.strip()): return s.strip()
    return None

def quadrant(v, e):
    if v >= 0.5 and e >= 0.5: return "happy / excited"
    if v >= 0.5 and e <  0.5: return "peaceful / content"
    if v <  0.5 and e >= 0.5: return "angry / aggressive"
    return "sad / melancholic"

def fetch_spotify_metadata(track_id):
    """Scrape title+artist from open.spotify.com og: tags. Needs network."""
    url = f"https://open.spotify.com/track/{track_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            page = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    title = re.search(r'<meta property="og:title" content="([^"]+)"', page)
    desc  = re.search(r'<meta property="og:description" content="([^"]+)"', page)
    if not title or not desc: return None
    title_text = html.unescape(title.group(1))
    desc_text  = html.unescape(desc.group(1))
    parts = [p.strip() for p in re.split(r"\u00b7|\\|", desc_text) if p.strip()]
    artist = parts[1] if len(parts) >= 2 else ""
    return {"title": title_text, "artist": artist}

def _tokenize(s):
    s = re.sub(r"[\s\W_]+", " ", s.lower()).strip()
    return [t for t in s.split() if len(t) > 1]

def fuzzy_catalog_lookup(query):
    """Match free-form 'title by artist' (or 'title - artist'). Requires ALL
    title tokens AND at least one artist token to match."""
    catalog = load_catalog()
    title_q, artist_q = None, None
    if re.search(r"\s+by\s+", query, re.I):
        a, b = re.split(r"\s+by\s+", query, maxsplit=1, flags=re.I)
        title_q, artist_q = a, b
    elif " - " in query:
        a, b = query.split(" - ", 1)
        title_q, artist_q = a, b
    title_tokens  = _tokenize(title_q  or query)
    artist_tokens = _tokenize(artist_q or "")
    if not title_tokens: return None
    titles  = catalog["track_name"].fillna("").str.lower().values
    artists = catalog["artists"].fillna("").str.lower().values
    title_hits = np.ones(len(catalog), dtype=bool)
    for t in title_tokens:
        title_hits &= np.array([t in name for name in titles])
    if not title_hits.any(): return None
    cand_idx = np.where(title_hits)[0]
    if artist_tokens:
        ok = []
        for i in cand_idx:
            blob = artists[i]
            if sum(t in blob for t in artist_tokens) >= 1:
                ok.append(i)
        if not ok: return None
        return catalog.iloc[int(ok[0])]
    return catalog.iloc[int(cand_idx[0])]

# ---------------- Claude integration ----------------
def claude_call(prompt, max_tokens=600, system=None):
    if USE_MOCK_AI:
        return _mock_claude(prompt)
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=max_tokens,
        system=system or "You are a thoughtful music analyst.",
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text

def _mock_claude(prompt):
    if "estimate" in prompt.lower() and "json" in prompt.lower():
        return json.dumps({
            "danceability": 0.7, "energy": 0.78, "loudness": -5.2,
            "speechiness": 0.05, "acousticness": 0.10, "instrumentalness": 0.001,
            "liveness": 0.10, "tempo": 105.0, "key": 1, "mode": 1,
            "time_signature": 4, "duration_ms": 175000,
            "explicit": False, "popularity": 80, "track_genre": "pop",
        })
    return ("[mock description] This song sits where pop confidence meets a hint "
            "of restless energy \u2014 upbeat enough to dance to, knowing enough "
            "to brood over.\n\n"
            "[mock match rationale] Suggested as a match because it lives in the "
            "same upper-mid valence / high-energy corner.\n\n"
            "[mock contrast rationale] Suggested as a contrast because it sits "
            "diagonally opposite, calmer and bluer.")

def claude_estimate_features(artist, title):
    prompt = (
        f"Estimate Spotify audio features for \"{title}\" by {artist}.\n\n"
        "Output ONLY a JSON object with these fields, no commentary:\n"
        "  danceability      : 0.0-1.0\n"
        "  energy            : 0.0-1.0\n"
        "  loudness          : dB, typically -25 to 0\n"
        "  speechiness       : 0.0-1.0\n"
        "  acousticness      : 0.0-1.0\n"
        "  instrumentalness  : 0.0-1.0 (1.0 = no vocals)\n"
        "  liveness          : 0.0-1.0\n"
        "  tempo             : BPM\n"
        "  key               : integer 0-11\n"
        "  mode              : 0=minor, 1=major\n"
        "  time_signature    : 3, 4, or 5\n"
        "  duration_ms       : integer\n"
        "  explicit          : true or false\n"
        "  popularity        : 0-100\n"
        "  track_genre       : pick from standard 114 Spotify genres (lowercase)"
    )
    text = claude_call(prompt, max_tokens=400,
        system="You estimate Spotify audio features. Output only valid JSON.")
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m: text = m.group(0)
    return json.loads(text)

def claude_describe_mood(artist, title, v, e, match_rec, contrast_rec):
    prompt = (
        f"\"{title}\" by {artist} maps to valence={v:.2f}, energy={e:.2f} "
        f"(quadrant: \"{quadrant(v, e)}\").\n\n"
        "Write three short paragraphs:\n"
        "1) Two sentences describing the mood, written for a music-app user.\n"
        f"2) One sentence on why \"{match_rec['title']}\" by {match_rec['artist']} "
        "is a good mood-MATCH recommendation.\n"
        f"3) One sentence on why \"{contrast_rec['title']}\" by {contrast_rec['artist']} "
        "is a good mood-CONTRAST recommendation.\n"
        "Plain text, blank line between paragraphs.")
    return claude_call(prompt, max_tokens=350)

# ---------------- Feature pipeline ----------------
def features_dict_to_X(feats, artist):
    load_models()
    f = dict(feats)
    f["explicit"] = int(bool(f.get("explicit", False)))
    f["log1p_duration"] = float(np.log1p(f["duration_ms"]))
    g = _genre_enc[_genre_enc["track_genre"] == f.get("track_genre", "")]
    if len(g) == 0:
        f["genre_valence_mean"] = _artist_enc["global_v"]
        f["genre_energy_mean"]  = _artist_enc["global_e"]
        f["genre_freq"] = 0
    else:
        f["genre_valence_mean"] = float(g.iloc[0]["genre_valence_mean"])
        f["genre_energy_mean"]  = float(g.iloc[0]["genre_energy_mean"])
        f["genre_freq"]         = float(g.iloc[0]["genre_freq"])
    SMOOTH = _artist_enc["smooth_alpha"]
    GV = _artist_enc["global_v"]; GE = _artist_enc["global_e"]
    a_first = (artist or "").split(";")[0].strip()
    iv = _artist_enc["valence"].get(a_first); ie = _artist_enc["energy"].get(a_first)
    f["artist_v_mean"] = GV if iv is None else (iv["s"] + SMOOTH * GV) / (iv["n"] + SMOOTH)
    f["artist_e_mean"] = GE if ie is None else (ie["s"] + SMOOTH * GE) / (ie["n"] + SMOOTH)
    key = int(f.get("key", 0))
    ts  = int(f.get("time_signature", 4))
    for i in range(12): f[f"key_{i}"] = int(i == key)
    for i in [1, 3, 4, 5]: f[f"ts_{i}"] = int(i == ts)
    for k in ("key", "time_signature", "track_genre"): f.pop(k, None)
    row = {c: f.get(c, 0) for c in _feature_cols}
    return pd.DataFrame([row])

def predict_from_features(feats, artist):
    load_models()
    X = features_dict_to_X(feats, artist)
    pv = float(np.clip(_models["valence"].predict(X)[0], 0, 1))
    pe = float(np.clip(_models["energy"].predict(X)[0],  0, 1))
    return pv, pe

def find_recommendations(pv, pe, exclude_artist=None, exclude_track_id=None):
    catalog = load_catalog()
    target = np.array([[pv, pe]])
    _, idxs = _nn_index.kneighbors(target, n_neighbors=20)
    match = None
    for i in idxs[0]:
        c = catalog.iloc[i]
        if exclude_track_id and c["track_id"] == exclude_track_id: continue
        if exclude_artist and c["artists"] == exclude_artist: continue
        match = c; break
    contrast_target = np.array([[1.0 - pv, 1.0 - pe]])
    _, c_idxs = _nn_index.kneighbors(contrast_target, n_neighbors=20)
    contrast = None
    for i in c_idxs[0]:
        c = catalog.iloc[i]
        if exclude_track_id and c["track_id"] == exclude_track_id: continue
        if exclude_artist and c["artists"] == exclude_artist: continue
        contrast = c; break
    return match, contrast

# ---------------- Public entry ----------------
def predict(query):
    catalog = load_catalog()
    track_id = extract_track_id(query)

    if track_id:
        hit = catalog[catalog["track_id"] == track_id]
        if len(hit):
            row = hit.iloc[0]
            pv, pe = float(row["pred_valence"]), float(row["pred_energy"])
            mr, cr = find_recommendations(pv, pe, row["artists"], row["track_id"])
            desc = claude_describe_mood(
                row["artists"], row["track_name"], pv, pe,
                {"artist": mr["artists"], "title": mr["track_name"]},
                {"artist": cr["artists"], "title": cr["track_name"]})
            return _result(row["artists"], row["track_name"], row["track_genre"],
                           pv, pe, mr, cr, desc, "catalog (track_id match)",
                           ground_truth={"valence": float(row["valence_true"]),
                                         "energy":  float(row["energy_true"])})
        meta = fetch_spotify_metadata(track_id)
        if not meta:
            return {"error": f"Track {track_id} not in our 89.5k catalog and "
                             "we couldn't reach open.spotify.com to look it up. "
                             "Pass 'artist - title' instead."}
        return _ai_path(meta["artist"], meta["title"])

    fuzzy = fuzzy_catalog_lookup(query)
    if fuzzy is not None:
        pv, pe = float(fuzzy["pred_valence"]), float(fuzzy["pred_energy"])
        mr, cr = find_recommendations(pv, pe, fuzzy["artists"], fuzzy["track_id"])
        desc = claude_describe_mood(
            fuzzy["artists"], fuzzy["track_name"], pv, pe,
            {"artist": mr["artists"], "title": mr["track_name"]},
            {"artist": cr["artists"], "title": cr["track_name"]})
        return _result(fuzzy["artists"], fuzzy["track_name"], fuzzy["track_genre"],
                       pv, pe, mr, cr, desc, "catalog (fuzzy text match)",
                       ground_truth={"valence": float(fuzzy["valence_true"]),
                                     "energy":  float(fuzzy["energy_true"])})

    artist, title = "", query
    if re.search(r"\s+by\s+", query, re.I):
        parts = re.split(r"\s+by\s+", query, maxsplit=1, flags=re.I)
        title, artist = parts[0].strip(), parts[1].strip()
    elif " - " in query:
        a, b = query.split(" - ", 1)
        title, artist = a.strip(), b.strip()
    return _ai_path(artist, title)

def _ai_path(artist, title):
    feats = claude_estimate_features(artist, title)
    pv, pe = predict_from_features(feats, artist)
    mr, cr = find_recommendations(pv, pe, exclude_artist=artist)
    desc = claude_describe_mood(
        artist, title, pv, pe,
        {"artist": mr["artists"], "title": mr["track_name"]},
        {"artist": cr["artists"], "title": cr["track_name"]})
    return _result(artist, title, feats.get("track_genre", "?"),
                   pv, pe, mr, cr, desc, "AI-estimated features",
                   estimated_features=feats)

def _result(artist, title, genre, pv, pe, mr, cr, desc, source,
            ground_truth=None, estimated_features=None):
    out = {
        "artist": artist, "title": title, "genre": genre,
        "predicted": {"valence": round(pv, 3), "energy": round(pe, 3),
                      "quadrant": quadrant(pv, pe)},
        "match_recommendation": {
            "artist": mr["artists"], "title": mr["track_name"],
            "valence": round(float(mr["pred_valence"]), 3),
            "energy":  round(float(mr["pred_energy"]),  3)},
        "contrast_recommendation": {
            "artist": cr["artists"], "title": cr["track_name"],
            "valence": round(float(cr["pred_valence"]), 3),
            "energy":  round(float(cr["pred_energy"]),  3)},
        "description": desc,
        "source": source,
    }
    if ground_truth is not None:
        out["ground_truth"] = {k: round(v, 3) for k, v in ground_truth.items()}
    if estimated_features is not None:
        out["estimated_features"] = estimated_features
    return out

def pretty_print(result):
    if "error" in result:
        print(f"\nError: {result['error']}\n"); return
    p = result["predicted"]
    print(f"\nYou picked: \"{result['title']}\" by {result['artist']}")
    print(f"  genre: {result['genre']}")
    print(f"  source: {result['source']}")
    print(f"\nPredicted mood")
    print(f"  valence: {p['valence']}")
    print(f"  energy:  {p['energy']}")
    print(f"  quadrant: {p['quadrant']}")
    if "ground_truth" in result:
        gt = result["ground_truth"]
        print(f"  (Spotify true values: valence={gt['valence']}, energy={gt['energy']})")
    if "estimated_features" in result:
        ef = result["estimated_features"]
        print(f"  AI-estimated audio features: {ef}")
    m = result["match_recommendation"]; c = result["contrast_recommendation"]
    print(f"\nMood-match recommendation:")
    print(f"  \"{m['title']}\" by {m['artist']}  (v={m['valence']}, e={m['energy']})")
    print(f"\nMood-contrast recommendation:")
    print(f"  \"{c['title']}\" by {c['artist']}  (v={c['valence']}, e={c['energy']})")
    print(f"\nDescription (Claude):\n{result['description']}\n")

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 \
            else "https://open.spotify.com/track/23PvWFdi76vER4p1e2Xroj"
    pretty_print(predict(query))
