"""Streamlit web UI for the Mood Predictor.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""
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
        help=("Used for input classification, audio-feature estimation, mood "
              "interpretation, and writing descriptions. Without a key the app "
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
    if st.button("\U0001F3B6  Espresso (AI estimate)"):
        st.session_state["query"] = "Espresso by Sabrina Carpenter"
    if st.button("\U0001F61E  \"I'm drained from studying\""):
        st.session_state["query"] = "I'm drained from reading these assignments"
    if st.button("\U0001F389  \"feeling celebratory\""):
        st.session_state["query"] = "I'm feeling celebratory after acing my exam"

# ---- Header ----
st.title("\U0001F3B5 Mood Predictor")
st.caption("Three things in one box: paste a Spotify URL, type \"Title by Artist\", "
           "or describe how you feel \u2014 the app figures out which.")

@st.cache_resource
def _warm_up():
    engine.load_catalog()
    engine.load_models()
    return True
_warm_up()

# ---- Input ----
default_query = st.session_state.get("query", "")
query = st.text_input(
    "Your input",
    value=default_query,
    placeholder="\"Mr. Brightside by The Killers\"  /  Spotify URL  /  \"I'm drained from studying\"",
)
go = st.button("Go", type="primary")

# ---- Helper: 2D plot ----
def plot_mood_space(highlight_points):
    """highlight_points: list of (label, valence, energy, color, marker, size)"""
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
    ax.set_xlabel("valence  (sad \u2190\u2192 happy)")
    ax.set_ylabel("energy   (calm \u2190\u2192 intense)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    return fig

# ---- Main flow ----
if go and query.strip():
    with st.spinner("Routing your input through the pipeline..."):
        try:
            result = engine.predict(query.strip())
        except Exception as e:
            st.error(f"Something went wrong: {type(e).__name__}: {e}")
            st.stop()

    # ----- Error / unparseable -----
    if result.get("kind") == "error" or "error" in result:
        st.warning(result["error"])
        st.stop()

    # =================================================================
    # MOOD RESULT (user described how they feel)
    # =================================================================
    if result["kind"] == "mood":
        t = result["target"]
        st.subheader("\U0001F4A4 You described a mood")
        st.write(f"**Your input:** \u201c{result['mood_text']}\u201d")
        if result.get("interpreted_summary"):
            st.caption(f"Interpreted as: *{result['interpreted_summary']}*")

        c1, c2, c3 = st.columns(3)
        c1.metric("Target valence", f"{t['valence']:.2f}")
        c2.metric("Target energy",  f"{t['energy']:.2f}")
        c3.metric("Quadrant", t["quadrant"])

        st.subheader("Where your mood lands in the 2D plane")
        highlights = [("your mood", t["valence"], t["energy"], "#264653", "*", 240)]
        for i, r in enumerate(result["recommendations"]):
            highlights.append(
                (f"rec #{i+1}: {r['title']}", r["valence"], r["energy"],
                 ["#2A9D8F", "#E9C46A", "#F4A261"][i % 3], "o", 130))
        st.pyplot(plot_mood_space(highlights))

        st.subheader("Songs picked for this mood")
        for r in result["recommendations"]:
            st.markdown(
                f"- **\u201c{r['title']}\u201d** by *{r['artist']}*  "
                f"<span style='color:#777'>(v={r['valence']}, e={r['energy']})</span>",
                unsafe_allow_html=True)

        st.subheader("Why these picks")
        st.write(result["description"])

        with st.expander("How this prediction was made"):
            st.markdown(
                "1. **Claude classified** your input as a mood description (not a song)\n"
                "2. **Claude mapped** the mood text to a target (valence, energy) "
                f"= ({t['valence']}, {t['energy']}) on the Russell circumplex\n"
                "3. **NearestNeighbors** over the catalog's predicted-(v, e) plane "
                "(every entry produced by your XGBoost) returned the closest 3 songs\n"
                "4. **Claude wrote** the rationale paragraph above"
            )

    # =================================================================
    # SONG RESULT (user gave a URL, song name, etc.)
    # =================================================================
    elif result["kind"] == "song":
        p = result["predicted"]
        col_song, col_mood = st.columns([1.1, 1])
        with col_song:
            st.subheader(f"\u201c{result['title']}\u201d")
            st.write(f"by **{result['artist']}**  ·  genre: *{result['genre']}*")
            if result["source"] == "catalog":
                st.caption("\u2705 Found in catalog \u2014 prediction was pre-computed by your XGBoost model")
            else:
                st.caption("\U0001F916 Not in catalog \u2014 Claude estimated audio features, your XGBoost predicted the mood")
        with col_mood:
            m1, m2 = st.columns(2)
            m1.metric("Valence (sad \u2194 happy)",  f"{p['valence']:.3f}")
            m2.metric("Energy (calm \u2194 intense)", f"{p['energy']:.3f}")
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

        st.subheader("Recommendations")
        rec_a, rec_b = st.columns(2)
        if result.get("match_recommendation"):
            mr = result["match_recommendation"]
            with rec_a:
                st.markdown("**\U0001F7E2 Mood match \u2014 lean in**")
                st.markdown(f"\u201c**{mr['title']}**\u201d  by *{mr['artist']}*")
                st.caption(f"valence {mr['valence']}  ·  energy {mr['energy']}")
        if result.get("contrast_recommendation"):
            cr = result["contrast_recommendation"]
            with rec_b:
                st.markdown("**\U0001F7E0 Mood contrast \u2014 pull yourself out**")
                st.markdown(f"\u201c**{cr['title']}**\u201d  by *{cr['artist']}*")
                st.caption(f"valence {cr['valence']}  ·  energy {cr['energy']}")

        st.subheader("How this song feels")
        st.write(result["description"])

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
                    "5. **Claude wrote** the description"
                )
            else:
                st.markdown(
                    f"1. **Input parsed** as `\"{result['title']}\" by {result['artist']}`\n"
                    "2. Not found in the catalog\n"
                    "3. **Claude estimated** 14 raw audio features\n"
                    "4. Same engineered feature pipeline as training (genre encoder, "
                    "artist encoder, key/ts one-hots, log1p_duration)\n"
                    "5. **Your XGBoost models** ran on those features \u2192 (valence, energy)\n"
                    "6. **NearestNeighbors** picked match + contrast\n"
                    "7. **Claude wrote** the description"
                )

else:
    st.info("Paste anything above to get started, or click an example in the sidebar. "
            "The app accepts Spotify URLs, song names, or mood descriptions.")
