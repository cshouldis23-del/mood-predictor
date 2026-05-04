"""Mood prediction from a song reference (Spotify URL or "Title by Artist").

Catalog hit -> stored XGBoost prediction. Catalog miss -> Claude estimates
audio features, XGBoost predicts mood. Either way, Claude writes a mood
description and rationales for the recommendations.
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
    _nn_index = NearestNeighbors(n_neighbors=200, algorithm="ball_tree").fit(
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
    parts = [p.strip() for p in re.split(r"·|\\|", desc_text) if p.strip()]
    artist = parts[1] if len(parts) >= 2 else ""
    return {"title": title_text, "artist": artist}

def _tokenize(s):
    s = re.sub(r"[\s\W_]+", " ", s.lower()).strip()
    return [t for t in s.split() if len(t) > 1]

_search_blobs = None  # populated lazily

def _get_search_blobs():
    """Cache a lowercase 'title artist' string per catalog row for fuzzy search."""
    global _search_blobs
    if _search_blobs is None:
        catalog = load_catalog()
        first_artist = catalog["artists"].fillna("").astype(str).str.split(";").str[0]
        blobs = (catalog["track_name"].fillna("").astype(str) + " " + first_artist)
        _search_blobs = blobs.str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.strip().values
    return _search_blobs

def fuzzy_catalog_lookup(query):
    """Match free-form queries to the catalog. Handles:
      - typos / misspellings (Levenshtein-style)
      - missing 'by'/'-' separators
      - reversed token order ('olivia rodrigo drivers license')
      - extra/missing punctuation
    Returns None if no candidate scores above the threshold."""
    from rapidfuzz import process, fuzz
    catalog = load_catalog()
    blobs = _get_search_blobs()

    # Strip common separators so 'Title by Artist' and 'Title - Artist' both
    # collapse into a clean 'title artist' phrase
    q = re.sub(r"\s+by\s+", " ", query, flags=re.I)
    q = q.replace(" - ", " ")
    q = re.sub(r"[^\w\s]", " ", q).strip().lower()
    if not q:
        return None

    # WRatio = best general-purpose scorer; tolerates word reordering, typos,
    # partial matches. score_cutoff filters out non-matches.
    # Get top-15 candidates, then break ties by popularity so common queries
    # like "drivers license" go to Olivia Rodrigo's original instead of an
    # obscure cover with the same title.
    # token_set_ratio: based on the proportion of overlapping tokens
    # between query and candidate. Stricter than WRatio (which over-rewards
    # single-word substring matches like "Benson Boone" -> "George Benson").
    # Cutoff 80 keeps typos in but rejects weak overlaps.
    candidates = process.extract(q, blobs, scorer=fuzz.token_set_ratio,
                                  limit=20, score_cutoff=80)
    if not candidates:
        return None
    # Additional sanity check: at least half of the query's significant tokens
    # (length > 2) must appear (substring) in the candidate blob. This blocks
    # weird matches that token_set_ratio still lets through.
    q_tokens = [t for t in q.split() if len(t) > 2]
    if q_tokens:
        def overlap(blob):
            hits = sum(1 for t in q_tokens if t in blob)
            return hits / len(q_tokens)
        candidates = [c for c in candidates if overlap(c[0]) >= 0.5]
        if not candidates:
            return None
    # Among the top-scoring batch (score within 5 of best), pick highest popularity
    top_score = candidates[0][1]
    short_list = [c for c in candidates if c[1] >= top_score - 5]
    has_pop = "popularity" in catalog.columns
    if has_pop and len(short_list) > 1:
        short_list.sort(key=lambda c: (-int(catalog.iloc[c[2]].get("popularity", 0)),
                                        -c[1]))
    return catalog.iloc[int(short_list[0][2])]

# ---------------- Claude ----------------
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
    p = prompt.lower()
    if "estimate" in p and "json" in p:
        return json.dumps({
            "danceability": 0.7, "energy": 0.78, "loudness": -5.2,
            "speechiness": 0.05, "acousticness": 0.10, "instrumentalness": 0.001,
            "liveness": 0.10, "tempo": 105.0, "key": 1, "mode": 1,
            "time_signature": 4, "duration_ms": 175000,
            "explicit": False, "popularity": 80, "track_genre": "pop",
        })
    return ("[mock description] This song sits in a particular corner of the mood "
            "plane.\n\n[mock match rationale] Same emotional zone.\n\n"
            "[mock contrast rationale] Diagonally opposite in mood.")

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
        "  track_genre       : pick from standard 114 Spotify genres (lowercase)")
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
        "Write three short paragraphs (plain text, blank line between):\n"
        "1) Two sentences describing the mood, written for a music-app user.\n"
        f"2) One sentence on why \"{match_rec['title']}\" by {match_rec['artist']} "
        "is a good mood-MATCH recommendation.\n"
        f"3) One sentence on why \"{contrast_rec['title']}\" by {contrast_rec['artist']} "
        "is a good mood-CONTRAST recommendation.")
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

MIN_POPULARITY = 30  # filter recommendations so they're recognizable songs

def find_recommendations(pv, pe, exclude_artist=None, exclude_track_id=None,
                        k=1, min_popularity=MIN_POPULARITY):
    """Search the catalog for the k nearest neighbors in (v, e) space,
    filtering for popularity >= min_popularity. Falls back to relaxing the
    popularity filter if fewer than k recognizable candidates exist."""
    catalog = load_catalog()
    target = np.array([[pv, pe]])
    _, idxs = _nn_index.kneighbors(target, n_neighbors=200)
    has_pop = "popularity" in catalog.columns
    # First pass: popularity filter on
    out = []
    for i in idxs[0]:
        c = catalog.iloc[i]
        if exclude_track_id and c["track_id"] == exclude_track_id: continue
        if exclude_artist and c["artists"] == exclude_artist: continue
        if has_pop and c["popularity"] < min_popularity: continue
        if any(c["track_name"] == x["track_name"] and c["artists"] == x["artists"]
               for x in out):
            continue
        out.append(c)
        if len(out) >= k: break
    # Fallback: if too few, relax the popularity filter
    if len(out) < k:
        for i in idxs[0]:
            c = catalog.iloc[i]
            if exclude_track_id and c["track_id"] == exclude_track_id: continue
            if exclude_artist and c["artists"] == exclude_artist: continue
            if any(c["track_id"] == x["track_id"] for x in out): continue
            out.append(c)
            if len(out) >= k: break
    return out

# ---------------- Public entry ----------------
def predict(query):
    catalog = load_catalog()
    q = (query or "").strip()
    if not q:
        return {"kind": "error", "error": "Please enter a song name or Spotify URL."}

    track_id = extract_track_id(q)
    if track_id:
        hit = catalog[catalog["track_id"] == track_id]
        if len(hit):
            return _build_song_result_from_catalog(hit.iloc[0])
        meta = fetch_spotify_metadata(track_id)
        if not meta:
            return {"kind": "error",
                    "error": f"Track {track_id} isn't in our 89.5k catalog and we "
                             "couldn't reach open.spotify.com to look it up. Try "
                             "typing the song's title and artist instead, like "
                             "\"Mr. Brightside by The Killers\"."}
        return _ai_song_path(meta["artist"], meta["title"])

    fuzzy = fuzzy_catalog_lookup(q)
    if fuzzy is not None:
        return _build_song_result_from_catalog(fuzzy)

    # Catalog miss -- route to AI feature estimation. Try to split into
    # title + artist if a 'by' or ' - ' separator is present; otherwise
    # pass the whole string as the title and let Claude figure it out.
    title, artist = q, ""
    if re.search(r"\s+by\s+", q, re.I):
        t, a = re.split(r"\s+by\s+", q, maxsplit=1, flags=re.I)
        title, artist = t.strip(), a.strip()
    elif " - " in q:
        a, b = q.split(" - ", 1)
        title, artist = a.strip(), b.strip()
    return _ai_song_path(artist, title)

def _build_song_result_from_catalog(row):
    pv, pe = float(row["pred_valence"]), float(row["pred_energy"])
    matches  = find_recommendations(pv, pe, row["artists"], row["track_id"], k=1)
    contrasts = find_recommendations(1 - pv, 1 - pe, row["artists"], row["track_id"], k=1)
    mr = matches[0] if matches else None
    cr = contrasts[0] if contrasts else None
    desc = claude_describe_mood(
        row["artists"], row["track_name"], pv, pe,
        {"artist": mr["artists"], "title": mr["track_name"]} if mr is not None else {"artist":"","title":""},
        {"artist": cr["artists"], "title": cr["track_name"]} if cr is not None else {"artist":"","title":""})
    return {
        "kind": "song",
        "source": "catalog",
        "artist": row["artists"], "title": row["track_name"], "genre": row["track_genre"],
        "predicted": {"valence": round(pv, 3), "energy": round(pe, 3),
                      "quadrant": quadrant(pv, pe)},
        "ground_truth": {"valence": round(float(row["valence_true"]), 3),
                         "energy":  round(float(row["energy_true"]),  3)},
        "match_recommendation":    _row_to_rec(mr) if mr is not None else None,
        "contrast_recommendation": _row_to_rec(cr) if cr is not None else None,
        "description": desc,
    }

def _ai_song_path(artist, title):
    if not (artist or title):
        return {"kind": "error", "error": "I couldn't extract an artist or title."}
    feats = claude_estimate_features(artist, title)
    pv, pe = predict_from_features(feats, artist)
    matches  = find_recommendations(pv, pe, exclude_artist=artist, k=1)
    contrasts = find_recommendations(1 - pv, 1 - pe, exclude_artist=artist, k=1)
    mr = matches[0] if matches else None
    cr = contrasts[0] if contrasts else None
    desc = claude_describe_mood(
        artist, title, pv, pe,
        {"artist": mr["artists"], "title": mr["track_name"]} if mr is not None else {"artist":"","title":""},
        {"artist": cr["artists"], "title": cr["track_name"]} if cr is not None else {"artist":"","title":""})
    return {
        "kind": "song",
        "source": "AI-estimated features",
        "artist": artist, "title": title, "genre": feats.get("track_genre", "?"),
        "predicted": {"valence": round(pv, 3), "energy": round(pe, 3),
                      "quadrant": quadrant(pv, pe)},
        "estimated_features": feats,
        "match_recommendation":    _row_to_rec(mr) if mr is not None else None,
        "contrast_recommendation": _row_to_rec(cr) if cr is not None else None,
        "description": desc,
    }

def _row_to_rec(r):
    return {"artist": r["artists"], "title": r["track_name"],
            "valence": round(float(r["pred_valence"]), 3),
            "energy":  round(float(r["pred_energy"]),  3),
            "popularity": int(r.get("popularity", 0))}

def pretty_print(result):
    if result.get("kind") == "error" or "error" in result:
        print(f"\n{result['error']}\n"); return
    p = result["predicted"]
    print(f"\nYou picked: \"{result['title']}\" by {result['artist']}")
    print(f"  genre: {result['genre']}    source: {result['source']}")
    print(f"\nPredicted mood: valence={p['valence']}  energy={p['energy']}  ({p['quadrant']})")
    if "ground_truth" in result:
        gt = result["ground_truth"]
        print(f"  (Spotify true: valence={gt['valence']}, energy={gt['energy']})")
    if "estimated_features" in result:
        print(f"  AI-estimated features: {result['estimated_features']}")
    m = result.get("match_recommendation"); c = result.get("contrast_recommendation")
    if m: print(f"\nMood-match: \"{m['title']}\" by {m['artist']}  (v={m['valence']}, e={m['energy']})")
    if c: print(f"Mood-contrast: \"{c['title']}\" by {c['artist']}  (v={c['valence']}, e={c['energy']})")
    print(f"\nDescription:\n{result['description']}\n")

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 \
            else "https://open.spotify.com/track/23PvWFdi76vER4p1e2Xroj"
    pretty_print(predict(query))
