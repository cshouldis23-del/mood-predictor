"""Streamlit web UI for the Mood Predictor."""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import predict_with_ai as engine

st.set_page_config(page_title="Mood Predictor", page_icon="\U0001F3B5", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    key_input = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help=("Used for audio-feature estimation on songs not in the catalog "
              "and for writing the mood description. Without a key the app "
              "uses built-in mock responses so you can still see the pipeline."))
    if key_input:
        os.environ["ANTHROPIC_API_KEY"] = key_input
        engine.USE_MOCK_AI = False
    else:
        engine.USE_MOCK_AI = True
    st.caption("AI mode: live Claude" if not engine.USE_MOCK_AI
               else "AI mode: mock (paste a key for live Claude)")

    st.divider()
    st.header("Try one of these")
    if st.button("\U0001F3B6  Mr. Brightside (catalog)"):
        st.session_state["query"] = "https://open.spotify.com/track/23PvWFdi76vER4p1e2Xroj"
    if st.button("\U0001F3B6  drivers license (catalog)"):
        st.session_state["query"] = "drivers license by Olivia Rodrigo"
    if st.button("\U0001F3B6  Africa (peaceful)"):
        st.session_state["query"] = "Africa by TOTO"
    if st.button("\U0001F3B6  Espresso (AI estimate)"):
        st.session_state["query"] = "Espresso by Sabrina Carpenter"

# ---- Header ----
st.title("\U0001F3B5 Mood Predictor")
st.write(
    "Would you like to receive some song suggestions that suit your current "
    "vibe? Drop in a Spotify song URL that's been in your head, or type a "
    "song and the artist that appeals to you in this moment. I will predict "
    "the song's mood with a description, then recommend a song you might "
    "also want to listen to as well as a song that is the opposite of your "
    "current mood — to change your day if you'd like.")

@st.cache_resource
def _warm_up():
    engine.load_catalog()
    engine.load_models()
    return True
_warm_up()

# ---- Input ----
default_query = st.session_state.get("query", "")
query = st.text_input(
    "Your song",
    value=default_query,
    placeholder="\"Mr. Brightside by The Killers\"  /  Spotify URL  /  \"Title - Artist\"",
)
go = st.button("Go", type="primary")

# ---- Helper: 2D plot ----
def plot_mood_space(highlight_points):
    sample = engine._catalog.iloc[
        np.random.RandomState(0).choice(len(engine._catalog), 8000, replace=False)]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(sample["pred_valence"], sample["pred_energy"],
               s=4, color="#cccccc", alpha=0.5, label="catalog (8k sample)")
    for label, v, e, color, marker, size in highlight_points:
        ax.scatter([v], [e], s=size, color=color, marker=marker,
                   edgecolor="white", linewidth=1.4, zorder=5, label=label)
    ax.axvline(0.5, color="black", lw=0.5, alpha=0.4)
    ax.axhline(0.5, color="black", lw=0.5, alpha=0.4)
    for x, y, lbl in [(0.06, 0.94, "angry / aggressive"),
                      (0.78, 0.94, "happy / excited"),
                      (0.06, 0.04, "sad / melancholic"),
                      (0.78, 0.04, "peaceful / content")]:
        ax.text(x, y, lbl, fontsize=9, color="#666", ha="left", va="center")
    ax.set_xlabel("valence  (sad ←→ happy)")
    ax.set_ylabel("energy   (calm ←→ intense)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    return fig

# ---- Main flow ----
if go and query.strip():
    with st.spinner("Looking up the track and running your XGBoost model..."):
        try:
            result = engine.predict(query.strip())
        except Exception as e:
            st.error(f"Something went wrong: {type(e).__name__}: {e}")
            st.stop()

    if result.get("kind") == "error" or "error" in result:
        st.warning(result["error"])
        st.stop()

    p = result["predicted"]
    col_song, col_mood = st.columns([1.1, 1])
    with col_song:
        st.subheader(f"“{result['title']}”")
        st.write(f"by **{result['artist']}**  ·  genre: *{result['genre']}*")
        if result["source"] == "catalog":
            st.caption("✅ Found in catalog — prediction was pre-computed by your XGBoost model")
        else:
            st.caption("\U0001F916 Not in catalog — Claude estimated audio features, your XGBoost predicted the mood")
    with col_mood:
        m1, m2 = st.columns(2)
        m1.metric("Valence (sad ↔ happy)",  f"{p['valence']:.3f}")
        m2.metric("Energy (calm ↔ intense)", f"{p['energy']:.3f}")
        st.markdown(f"**Mood quadrant:** {p['quadrant']}")
        if "ground_truth" in result:
            gt = result["ground_truth"]
            st.caption(f"Spotify's true values: valence={gt['valence']}, energy={gt['energy']}")

    st.subheader("Where the song lives in the 2D mood space")
    highlights = [
        (f"your song ({result['title']})", p["valence"], p["energy"], "#264653", "*", 240),
    ]
    if result.get("match_recommendation"):
        mr = result["match_recommendation"]
        highlights.append((f"match: {mr['title']}", mr["valence"], mr["energy"], "#2A9D8F", "o", 140))
    if result.get("contrast_recommendation"):
        cr = result["contrast_recommendation"]
        highlights.append((f"contrast: {cr['title']}", cr["valence"], cr["energy"], "#E76F51", "o", 140))
    st.pyplot(plot_mood_space(highlights))

    st.subheader("How this song feels")
    st.write(result["description"])

    st.subheader("Recommendations")
    rec_a, rec_b = st.columns(2)
    if result.get("match_recommendation"):
        mr = result["match_recommendation"]
        with rec_a:
            st.markdown("**\U0001F7E2 Mood match — lean in**")
            st.markdown(f"“**{mr['title']}**”  by *{mr['artist']}*")
            st.caption(f"valence {mr['valence']}  ·  energy {mr['energy']}")
    if result.get("contrast_recommendation"):
        cr = result["contrast_recommendation"]
        with rec_b:
            st.markdown("**\U0001F7E0 Mood contrast — pull yourself out**")
            st.markdown(f"“**{cr['title']}**”  by *{cr['artist']}*")
            st.caption(f"valence {cr['valence']}  ·  energy {cr['energy']}")

    if "estimated_features" in result:
        with st.expander("AI-estimated audio features (input to your XGBoost)"):
            ef = result["estimated_features"]
            st.dataframe(pd.DataFrame.from_records([ef]).T.rename(
                columns={0: "estimated"}), use_container_width=True)

    with st.expander("How this prediction was made"):
        if result["source"] == "catalog":
            st.markdown(
                "1. **Input parsed** as a track lookup (URL or song-name fuzzy match)\n"
                "2. **Catalog hit** in our pre-computed 89,579-track table\n"
                "3. The (valence, energy) shown was produced by **your XGBoost models** "
                "when we built the catalog\n"
                "4. **NearestNeighbors** picked match + contrast\n"
                "5. **Claude wrote** the mood description")
        else:
            st.markdown(
                f"1. **Input parsed** as `\"{result['title']}\" by {result['artist']}`\n"
                "2. Not found in the catalog\n"
                "3. **Claude estimated** 14 raw audio features\n"
                "4. Same engineered feature pipeline as training (genre encoder, "
                "artist encoder, key/ts one-hots, log1p_duration)\n"
                "5. **Your XGBoost models** ran on those features → (valence, energy)\n"
                "6. **NearestNeighbors** picked match + contrast\n"
                "7. **Claude wrote** the description")

else:
    st.info("Paste a Spotify URL or type a song name (e.g. \"Mr. Brightside by The Killers\") to get started, "
            "or click an example in the sidebar.")
