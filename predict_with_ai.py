"""Mood prediction with input routing.

Three input modes, auto-detected:
  1. Spotify URL or track ID            -> catalog lookup, else AI estimate
  2. "Title by Artist" / "Title - Artist" -> fuzzy catalog match, else AI estimate
  3. Free-form mood text                  -> mood -> target (v, e) -> recommendations

Garbage input returns a friendly error instead of fake predictions.
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

# --------- Claude ---------
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
    if "classify" in p:
        m = re.search(r'input:\s*"""(.+?)"""', prompt, re.DOTALL | re.IGNORECASE)
        ui = (m.group(1) if m else "").lower().strip()
        if len(ui) < 3:
            return '{"kind":"unparseable"}'
        if any(w in ui for w in [" by ", " - ", "spotify"]):
            return '{"kind":"song"}'
        if re.search(r"\bi('m| am|\s)|feeling|feel\s|tired|drained|happy|sad|"
                     r"angry|excited|stressed|exhausted|calm|content|joyful|"
                     r"melancho|anxious|lonely|burnt|burned|down\b|frustrat",
                     ui):
            return '{"kind":"mood"}'
        if re.search(r"[a-z]{3,}\s+[a-z]{3,}", ui):
            return '{"kind":"song"}'
        return '{"kind":"unparseable"}'
    if "estimate" in p and "json" in p:
        return json.dumps({
            "danceability": 0.7, "energy": 0.78, "loudness": -5.2,
            "speechiness": 0.05, "acousticness": 0.10, "instrumentalness": 0.001,
            "liveness": 0.10, "tempo": 105.0, "key": 1, "mode": 1,
            "time_signature": 4, "duration_ms": 175000,
            "explicit": False, "popularity": 80, "track_genre": "pop",
        })
    if "russell circumplex" in p and "summary" in p:
        return '{"valence":0.30,"energy":0.25,"summary":"low-energy and slightly down"}'
    return ("[mock description] This song sits in a particular corner of the mood "
            "plane.\n\n[mock match rationale] Same emotional zone.\n\n"
            "[mock contrast rationale] Diagonally opposite in mood.")

def claude_classify_input(query):
    prompt = (
        "Classify this user input for a music recommendation app. Output ONLY a "
        'JSON object {"kind":"song"|"mood"|"unparseable"}.\n\n'
        '- "song": refers to a specific song (a title, artist+title, Spotify URL, '
        'or commonly known song name)\n'
        '- "mood": describes how the person feels right now '
        '(e.g. "I\'m drained", "feeling happy", "stressed about exams")\n'
        '- "unparseable": gibberish, empty, or unrelated\n\n'
        f'Input: """{query}"""\n\nOutput:')
    text = claude_call(prompt, max_tokens=60,
                       system="You classify user input. Output only JSON.")
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m: return "unparseable"
    try:
        data = json.loads(m.group(0))
        kind = data.get("kind", "unparseable")
        if kind not in ("song", "mood", "unparseable"):
            return "unparseable"
        return kind
    except Exception:
        return "unparseable"

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

def claude_mood_to_coords(mood_text):
    prompt = (
        f"A user describes their mood as: \"{mood_text}\"\n\n"
        "Map this to coordinates on the Russell circumplex of affect. Output ONLY "
        'a JSON object: {"valence": 0.0-1.0, "energy": 0.0-1.0, "summary": "<one '
        'short phrase>"}\n\n'
        "Reference points:\n"
        '  "exhausted, drained" -> valence 0.30, energy 0.20\n'
        '  "sad, lonely"        -> valence 0.20, energy 0.35\n'
        '  "calm, content"      -> valence 0.65, energy 0.30\n'
        '  "joyful, ecstatic"   -> valence 0.85, energy 0.90\n'
        '  "angry, frustrated"  -> valence 0.20, energy 0.85\n'
        '  "anxious, stressed"  -> valence 0.30, energy 0.70')
    text = claude_call(prompt, max_tokens=120,
        system="You map mood to Russell circumplex coords. Output only JSON.")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: raise ValueError("Couldn't parse mood JSON")
    return json.loads(m.group(0))

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

def claude_describe_mood_recs(mood_text, v, e, recs):
    rec_list = "\n".join(
        f"  - \"{r['title']}\" by {r['artist']} (v={r['valence']}, e={r['energy']})"
        for r in recs[:3])
    prompt = (
        f"User's mood: \"{mood_text}\"\n"
        f"Mapped to valence={v:.2f}, energy={e:.2f} (quadrant: \"{quadrant(v, e)}\")\n"
        f"Recommended songs:\n{rec_list}\n\n"
        "Write a short, warm paragraph (3-4 sentences) explaining why these "
        "recommendations fit the user's mood. Plain text, no headers.")
    return claude_call(prompt, max_tokens=250)

# --------- Feature pipeline ---------
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

def find_recommendations(pv, pe, exclude_artist=None, exclude_track_id=None, k=3):
    catalog = load_catalog()
    target = np.array([[pv, pe]])
    _, idxs = _nn_index.kneighbors(target, n_neighbors=20)
    out = []
    for i in idxs[0]:
        c = catalog.iloc[i]
        if exclude_track_id and c["track_id"] == exclude_track_id: continue
        if exclude_artist and c["artists"] == exclude_artist: continue
        out.append(c)
        if len(out) >= k: break
    return out

# --------- Public entry ---------
def predict(query):
    catalog = load_catalog()
    q = (query or "").strip()
    if not q:
        return {"kind": "error", "error": "Please type something."}

    track_id = extract_track_id(q)
    if track_id:
        hit = catalog[catalog["track_id"] == track_id]
        if len(hit):
            return _build_song_result_from_catalog(hit.iloc[0])
        meta = fetch_spotify_metadata(track_id)
        if not meta:
            return {"kind": "error",
                    "error": f"Track {track_id} isn't in our 89.5k catalog and "
                             "we couldn't reach open.spotify.com to look it up. "
                             "Try typing the song's title and artist instead."}
        return _ai_song_path(meta["artist"], meta["title"])

    fuzzy = fuzzy_catalog_lookup(q)
    if fuzzy is not None:
        return _build_song_result_from_catalog(fuzzy)

    kind = claude_classify_input(q)
    if kind == "song":
        artist, title = "", q
        if re.search(r"\s+by\s+", q, re.I):
            t, a = re.split(r"\s+by\s+", q, maxsplit=1, flags=re.I)
            title, artist = t.strip(), a.strip()
        elif " - " in q:
            a, b = q.split(" - ", 1)
            title, artist = a.strip(), b.strip()
        return _ai_song_path(artist, title)
    if kind == "mood":
        return _mood_path(q)

    return {"kind": "error",
            "error": "I couldn't tell what you meant. Try one of these:\n"
                     "  - a Spotify track URL\n"
                     "  - a song name like \"Mr. Brightside by The Killers\"\n"
                     "  - how you feel like \"I'm exhausted from studying\""}

def _build_song_result_from_catalog(row):
    pv, pe = float(row["pred_valence"]), float(row["pred_energy"])
    recs = find_recommendations(pv, pe, row["artists"], row["track_id"], k=1)
    contrast = find_recommendations(1 - pv, 1 - pe, row["artists"], row["track_id"], k=1)
    mr = recs[0] if recs else None
    cr = contrast[0] if contrast else None
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
        "match_recommendation": _row_to_rec(mr) if mr is not None else None,
        "contrast_recommendation": _row_to_rec(cr) if cr is not None else None,
        "description": desc,
    }

def _ai_song_path(artist, title):
    if not (artist or title):
        return {"kind": "error", "error": "I couldn't extract an artist or title."}
    feats = claude_estimate_features(artist, title)
    pv, pe = predict_from_features(feats, artist)
    recs     = find_recommendations(pv, pe, exclude_artist=artist, k=1)
    contrast = find_recommendations(1 - pv, 1 - pe, exclude_artist=artist, k=1)
    mr = recs[0] if recs else None
    cr = contrast[0] if contrast else None
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
        "match_recommendation": _row_to_rec(mr) if mr is not None else None,
        "contrast_recommendation": _row_to_rec(cr) if cr is not None else None,
        "description": desc,
    }

def _mood_path(mood_text):
    coords = claude_mood_to_coords(mood_text)
    pv = float(np.clip(coords["valence"], 0, 1))
    pe = float(np.clip(coords["energy"],  0, 1))
    summary = coords.get("summary", "")
    recs = find_recommendations(pv, pe, k=3)
    desc = claude_describe_mood_recs(mood_text, pv, pe,
        [_row_to_rec(r) for r in recs])
    return {
        "kind": "mood",
        "mood_text": mood_text,
        "interpreted_summary": summary,
        "target": {"valence": round(pv, 3), "energy": round(pe, 3),
                   "quadrant": quadrant(pv, pe)},
        "recommendations": [_row_to_rec(r) for r in recs],
        "description": desc,
    }

def _row_to_rec(r):
    return {"artist": r["artists"], "title": r["track_name"],
            "valence": round(float(r["pred_valence"]), 3),
            "energy":  round(float(r["pred_energy"]),  3)}

def pretty_print(result):
    if result.get("kind") == "error" or "error" in result:
        print(f"\n{result['error']}\n"); return
    if result["kind"] == "mood":
        t = result["target"]
        print(f"\nYour mood: \"{result['mood_text']}\"")
        if result.get("interpreted_summary"):
            print(f"  interpreted as: {result['interpreted_summary']}")
        print(f"  mapped to valence={t['valence']}, energy={t['energy']} ({t['quadrant']})")
        print("\nRecommendations:")
        for r in result["recommendations"]:
            print(f"  \"{r['title']}\" by {r['artist']}  (v={r['valence']}, e={r['energy']})")
        print(f"\n{result['description']}\n")
        return
    p = result["predicted"]
    print(f"\nYou picked: \"{result['title']}\" by {result['artist']}")
    print(f"  genre: {result['genre']}    source: {result['source']}")
    print(f"\nPredicted mood")
    print(f"  valence: {p['valence']}    energy: {p['energy']}")
    print(f"  quadrant: {p['quadrant']}")
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
