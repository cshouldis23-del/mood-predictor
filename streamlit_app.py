"""
Streamlit web UI for the Mood Predictor.

Run locally:
    pip install streamlit xgboost scikit-learn pandas numpy matplotlib anthropic
    streamlit run streamlit_app.py

Deploy free on Streamlit Cloud:
    1. Push this folder to a GitHub repo
    2. Sign in at https://streamlit.io/cloud
    3. Pick the repo and set the entry point to streamlit_app.py
    4. In the app's "Secrets" panel, add:
           ANTHROPIC_API_KEY = "sk-ant-..."
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use the inference module from the same folder.
import predict_with_ai as engine

st.set_page_config(page_title="Mood Predictor",
                   page_icon="🎵",
                   layout="wide")

# ----- API key handling (sidebar) -----
with st.sidebar:
    st.header("Settings")
    key_input = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help=("Used to estimate audio features for songs not in the catalog "
              "and to write the mood description. Without a key, the app uses "
              "a built-in mock response so you can still see the pipeline."))
    if key_input:
        os.environ["ANTHROPIC_API_KEY"] = key_input
        engine.USE_MOCK_AI = False
    else:
        engine.USE_MOCK_AI = True
    st.caption(("AI mode: live Claude" if not engine.USE_MOCK_AI
                else "AI mode: mock (paste a key for live Claude)"))

    st.divider()
    st.header("Try one of these")
    if st.button("Mr. Brightside (catalog hit)"):
        st.session_state["query"] = "https://open.spotify.com/track/23PvWFdi76vER4p1e2Xroj"
    if st.button("drivers license (catalog hit)"):
        st.session_state["query"] = "drivers license by Olivia Rodrigo"
    if st.button("Take Me Out (catalog hit)"):
        st.session_state["query"] = "Take Me Out by Franz Ferdinand"
    if st.button("Espresso (catalog miss → AI path)"):
        st.session_state["query"] = "Espresso by Sabrina Carpenter"

# ----- Header -----
st.title("🎵 Mood Predictor")
st.caption("Predicts a song's valence (sad ↔ happy) and energy (calm ↔ intense), "
           "then suggests a mood-match and a mood-contrast track.")

# ----- Cache catalog + models -----
@st.cache_resource
def _warm_up():
    engine.load_catalog()
    engine.load_models()
    return True
_warm_up()

# ----- Input -----
default_query = st.session_state.get("query", "")
query = st.text_input(
    "Paste a Spotify URL or type \"Title by Artist\"",
    value=default_query,
    placeholder="https://open.spotify.com/track/...  or  Mr. Brightside by The Killers",
)
go = st.button("Predict mood", type="primary")

# ----- Main flow -----
if go and query.strip():
    with st.spinner("Looking up the track and running the model..."):
        try:
            result = engine.predict(query.strip())
        except Exception as e:
            st.error(f"Something went wrong: {type(e).__name__}: {e}")
            st.stop()

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # ---------- Top-of-page summary ----------
    p = result["predicted"]
    col_song, col_mood = st.columns([1.1, 1])
    with col_song:
        st.subheader(f"\u201c{result['title']}\u201d")
        st.write(f"by **{result['artist']}**  ·  genre: *{result['genre']}*")
        src = result["source"]
        if "track_id" in src:
            st.caption(f"\u2705 Found in catalog \u2014 prediction was pre-computed by the trained XGBoost model")
        elif "fuzzy" in src:
            st.caption(f"\u2705 Catalog match (fuzzy) \u2014 prediction is your XGBoost output for this song")
        else:
            st.caption(f"\U0001F916 Not in catalog \u2014 Claude estimated audio features, your XGBoost predicted the mood")

    with col_mood:
        m1, m2 = st.columns(2)
        m1.metric("Valence (sad ↔ happy)", f"{p['valence']:.3f}")
        m2.metric("Energy (calm ↔ intense)", f"{p['energy']:.3f}")
        st.markdown(f"**Mood quadrant:** {p['quadrant']}")
        if "ground_truth" in result:
            gt = result["ground_truth"]
            st.caption(f"Spotify's true values: valence={gt['valence']}, energy={gt['energy']}")

    # ---------- 2D mood-space plot ----------
    st.subheader("Where the song lives in the 2D mood space")
    catalog = engine._catalog
    sample_idx = np.random.RandomState(0).choice(len(catalog), 8000, replace=False)
    sample = catalog.iloc[sample_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(sample["pred_valence"], sample["pred_energy"],
               s=4, color="#cccccc", alpha=0.5, label="catalog (sample of 8k)")

    pv, pe = p["valence"], p["energy"]
    mr = result["match_recommendation"]
    cr = result["contrast_recommendation"]

    ax.scatter([pv], [pe], s=180, color="#264653", marker="*",
               edgecolor="white", linewidth=1.5, zorder=5,
               label=f"your song ({result['title']})")
    ax.scatter([mr["valence"]], [mr["energy"]], s=130, color="#2A9D8F",
               edgecolor="white", linewidth=1.2, zorder=4,
               label=f"match: {mr['title']}")
    ax.scatter([cr["valence"]], [cr["energy"]], s=130, color="#E76F51",
               edgecolor="white", linewidth=1.2, zorder=4,
               label=f"contrast: {cr['title']}")

    # Quadrant labels
    ax.axvline(0.5, color="black", linewidth=0.5, alpha=0.4)
    ax.axhline(0.5, color="black", linewidth=0.5, alpha=0.4)
    for x, y, label in [(0.06, 0.94, "angry / aggressive"),
                        (0.78, 0.94, "happy / excited"),
                        (0.06, 0.04, "sad / melancholic"),
                        (0.78, 0.04, "peaceful / content")]:
        ax.text(x, y, label, fontsize=9, color="#666",
                ha="left", va="center")

    ax.set_xlabel("valence  (sad ←→ happy)")
    ax.set_ylabel("energy   (calm ←→ intense)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    st.pyplot(fig)

    # ---------- Recommendations ----------
    st.subheader("Recommendations")
    rec_a, rec_b = st.columns(2)
    with rec_a:
        st.markdown("**🟢 Mood match — lean in**")
        st.markdown(f"\u201c**{mr['title']}**\u201d  by *{mr['artist']}*")
        st.caption(f"valence {mr['valence']}  ·  energy {mr['energy']}")
    with rec_b:
        st.markdown("**🟠 Mood contrast — pull yourself out**")
        st.markdown(f"\u201c**{cr['title']}**\u201d  by *{cr['artist']}*")
        st.caption(f"valence {cr['valence']}  ·  energy {cr['energy']}")

    # ---------- Claude description ----------
    st.subheader("How this song feels")
    st.write(result["description"])

    # ---------- Estimated features (AI path only) ----------
    if "estimated_features" in result:
        with st.expander("AI-estimated audio features (input to the XGBoost)"):
            ef = result["estimated_features"]
            df_show = pd.DataFrame.from_records([ef]).T
            df_show.columns = ["estimated"]
            st.dataframe(df_show, use_container_width=True)

    # ---------- Pipeline trace ----------
    with st.expander("How this prediction was made (pipeline trace)"):
        if "track_id" in result["source"] or "fuzzy" in result["source"]:
            st.markdown(f"""
1. **Input parsed** as track lookup
2. **Catalog hit** in our pre-computed 89,579-track table
3. The (valence, energy) shown was produced by your **XGBoost models** (`xgb_valence.json` and `xgb_energy.json`) when we built the catalog
4. **NearestNeighbors** over the catalog's predicted-(v, e) plane picked the match and contrast
5. **Claude** wrote the natural-language mood description
""")
        else:
            st.markdown(f"""
1. **Input parsed** as `\"{result['title']}\" by {result['artist']}`
2. Not found in the 89,579-track catalog
3. **Claude** estimated 14 raw audio features (danceability, energy, loudness, tempo, etc.)
4. Same engineered feature pipeline applied (genre encoder, artist encoder, key/ts one-hots, log1p_duration)
5. Your **XGBoost models** ran on those features → predicted (valence, energy)
6. **NearestNeighbors** over the catalog picked the match and contrast
7. **Claude** wrote the natural-language mood description
""")

else:
    st.info("Paste a Spotify track URL or type a song name above to get started, "
            "or click an example in the sidebar.")
