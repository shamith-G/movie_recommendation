import os
import json
import re
import pickle
import requests
import numpy as np
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TMDB_API_KEY   = os.getenv("TMDB_API_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
TMDB_BASE      = "https://api.themoviedb.org/3"
IMG_W500       = "https://image.tmdb.org/t/p/w500"
IMG_ORIG       = "https://image.tmdb.org/t/p/original"
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
api_key = st.secrets["TMDB_API_KEY"]

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineAI – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #050b14 !important; color: #e2e8f0; }
.stButton > button {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #fff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}
.stButton > button:hover {
    background: #7c3aed !important;
    border-color: #a78bfa !important;
}
.cast-img { border-radius: 12px; margin-bottom: 5px; border: 1px solid rgba(255,255,255,0.1); }
.rec-card { border-radius: 10px; transition: transform 0.2s; }
.rec-card:hover { transform: scale(1.05); }
</style>
""", unsafe_allow_html=True)

# ─── TF-IDF LOADING ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tfidf():
    with open(os.path.join(BASE_DIR, "df.pkl"), "rb") as f:
        df = pickle.load(f)
    with open(os.path.join(BASE_DIR, "indices.pkl"), "rb") as f:
        indices_obj = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    title_to_idx = {}
    if isinstance(indices_obj, dict):
        for k, v in indices_obj.items():
            title_to_idx[str(k).strip().lower()] = int(v)
    else:
        for k, v in indices_obj.items():
            title_to_idx[str(k).strip().lower()] = int(v)
    return df, tfidf_matrix, title_to_idx

def tfidf_recommend(title, top_n=12):
    df, tfidf_matrix, title_to_idx = load_tfidf()
    key = title.strip().lower()
    if key not in title_to_idx:
        return []
    idx = title_to_idx[key]
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    out = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            out.append(str(df.iloc[int(i)]["title"]))
        except Exception:
            continue
        if len(out) >= top_n:
            break
    return out

# rest of your code below...
def fetch_movies():
    url = f"https://api.themoviedb.org/3/movie/popular"
    params = {"api_key": api_key, "language": "en-US"}
    ...

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def img(path, size="w500"):
    base = IMG_W500 if size == "w500" else IMG_ORIG
    return f"{base}{path}" if path else None

@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_get(path, **params):
    params["api_key"] = TMDB_API_KEY
    params.setdefault("language", "en-US")
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=20)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_search_title(title):
    data = tmdb_get("/search/movie", query=title, page=1)
    results = data.get("results", [])
    return results[0] if results else None

# ─── MOVIE GRID RENDERER ──────────────────────────────────────────────────────
def render_grid(movies, prefix, cols=6):
    if not movies: return
    for i in range(0, len(movies), cols):
        columns = st.columns(cols)
        for j, m in enumerate(movies[i:i+cols]):
            with columns[j]:
                p_url = img(m.get("poster_path"))
                if p_url:
                    st.image(p_url, use_container_width=True)
                st.markdown(f"<p style='font-size:12px; font-weight:600; margin:0;'>{m.get('title','')[:25]}</p>", unsafe_allow_html=True)
                if st.button("Details", key=f"{prefix}_{m['id']}_{i}_{j}", use_container_width=True):
                    st.session_state["selected_id"] = m["id"]
                    st.rerun()

# ─── DETAIL VIEW (WITH CAST & RECOMMENDATIONS) ────────────────────────────────
def show_movie_detail(movie_id: int):
    # Get details, cast, and videos from TMDB
    d = tmdb_get(f"/movie/{movie_id}", append_to_response="credits,videos,recommendations")
    
    backdrop = img(d.get("backdrop_path"), size="original")
    poster = img(d.get("poster_path"))
    cast = (d.get("credits") or {}).get("cast", [])[:6]
    videos = (d.get("videos") or {}).get("results", [])
    
    trailer_url = next((f"https://youtube.com/watch?v={v['key']}" 
                       for v in videos if v['site'] == 'YouTube' and v['type'] == 'Trailer'), None)

    # Header / Banner
    st.markdown(f"""
        <div style="position:relative; height:380px; border-radius:20px; overflow:hidden; margin-bottom:-100px;">
            <img src="{backdrop}" style="width:100%; height:100%; object-fit:cover; filter: brightness(0.4);">
            <div style="position:absolute; inset:0; background: linear-gradient(0deg, #050b14 15%, transparent 100%);"></div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.image(poster, use_container_width=True)
    with c2:
        st.title(d.get("title"))
        st.markdown(f"*{d.get('tagline', '')}*")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Rating", f"⭐ {d.get('vote_average', 0):.1f}")
        m_col2.metric("Runtime", f"{d.get('runtime')} min")
        m_col3.metric("Year", d.get('release_date', '')[:4])

        b_col1, b_col2, b_col3 = st.columns(3)
        if trailer_url:
            b_col1.link_button("▶ Watch Trailer", trailer_url, use_container_width=True)
        if b_col2.button("✕ Back to Browse", use_container_width=True):
            st.session_state["selected_id"] = None
            st.rerun()
        
        is_fav = any(w['id'] == movie_id for w in st.session_state.get("watchlist", []))
        if b_col3.button("✓ In List" if is_fav else "+ Watchlist", use_container_width=True):
            if not is_fav:
                st.session_state["watchlist"].append({"id": movie_id, "title": d['title']})
                st.rerun()

    st.markdown("### 📖 Overview")
    st.write(d.get("overview"))

    # Cast Section
    if cast:
        st.markdown("### 🎭 Top Cast")
        cast_cols = st.columns(len(cast))
        for i, person in enumerate(cast):
            with cast_cols[i]:
                p_img = img(person.get("profile_path"))
                if p_img:
                    st.markdown(f'<img src="{p_img}" class="cast-img" style="width:100%">', unsafe_allow_html=True)
                st.caption(f"**{person['name']}**")
                st.markdown(f"<p style='font-size:11px; color:#94a3b8;'>{person['character']}</p>", unsafe_allow_html=True)

    # RECOMMENDATIONS — try local pickle TF-IDF first, fall back to TMDB API
    tfidf_titles = tfidf_recommend(d.get("title"), top_n=12)
    rec_results = []
    if tfidf_titles:
        for t in tfidf_titles:
            m = tmdb_search_title(t)
            if m:
                rec_results.append(m)
            if len(rec_results) >= 12:
                break
    else:
        # Movie not in local dataset — use TMDB's own recommendations
        tmdb_recs = (d.get("recommendations") or {}).get("results", [])
        rec_results = tmdb_recs[:12]

    if rec_results:
        st.divider()
        st.markdown("### 🎯 You Might Also Like")
        render_grid(rec_results, f"rec_{movie_id}", cols=6)

# ─── APP STATE & SIDEBAR ──────────────────────────────────────────────────────
if "watchlist" not in st.session_state: st.session_state["watchlist"] = []
if "selected_id" not in st.session_state: st.session_state["selected_id"] = None

with st.sidebar:
    st.title("🎬 CineAI")
    st.markdown("---")
    st.subheader(f"My Watchlist ({len(st.session_state['watchlist'])})")
    for item in st.session_state["watchlist"]:
        if st.button(f"🍿 {item['title'][:20]}", key=f"side_{item['id']}"):
            st.session_state["selected_id"] = item['id']
            st.rerun()

    st.markdown("---")
    st.markdown("[API Docs](https://movie-recommendation-hiil.onrender.com)")

# ─── HEADER ─────────────────────────────────────────────────────────────────
h1, h2 = st.columns([1.4, 6])
with h1:
    if st.button("🎬 **Cine-AI**", use_container_width=True):
        st.session_state["selected_id"] = None
        st.rerun()
with h2:
    st.markdown("##### Discover Your Next Favorite Movie")

# ─── MAIN ROUTER ──────────────────────────────────────────────────────────────
if st.session_state["selected_id"]:
    show_movie_detail(st.session_state["selected_id"])
else:
    tab1, tab2, tab3 = st.tabs(["🔥 Popular", "🔍 Search", "🌟 Trending"])
    
    with tab1:
        popular = tmdb_get("/movie/popular").get("results", [])
        render_grid(popular, "pop")
        
    with tab2:
        query = st.text_input("Search for any movie title...", placeholder="e.g. Interstellar")
        if query:
            results = tmdb_get("/search/movie", query=query).get("results", [])
            render_grid(results, "search")
            
    with tab3:
        trending = tmdb_get("/trending/movie/week").get("results", [])
        render_grid(trending, "trend")