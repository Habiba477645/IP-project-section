import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Movie Recommendation",  layout="wide")

st.markdown("""
<style>
/* Dark background */
.stApp { background-color: #0f0f1a; color: #e0e0f0; }

/* Sidebar-style left panel feel via columns */
h1 { font-size: 32px !important; color: #c084fc !important; letter-spacing: 2px; }
h2, h3 { color: #a78bfa !important; }

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div { background: #1e1e30 !important; color: #e0e0f0 !important; border: 1px solid #4c1d95 !important; border-radius: 8px; }

/* Slider */
div[data-testid="stSlider"] { accent-color: #a78bfa; }

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white; border: none; border-radius: 10px;
    padding: 12px 0; font-size: 16px; font-weight: 700;
    width: 100%; letter-spacing: 1px; cursor: pointer;
    transition: opacity .2s;
}
div.stButton > button:hover { opacity: 0.85; color: white; }

/* Tabs */
button[data-baseweb="tab"] { color: #a78bfa !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #7c3aed !important; color: #c084fc !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #1e1e30; border: 1px solid #4c1d95;
    border-radius: 12px; padding: 16px;
}

/* Movie cards */
.movie-card {
    background: #1a1a2e;
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.movie-info { flex: 1; }
.movie-num  { font-size: 11px; color: #6b7280; margin-bottom: 2px; }
.movie-title { font-size: 15px; font-weight: 700; color: #e0e0f0; }
.movie-genre { font-size: 12px; color: #7c6fcd; margin-top: 3px; }
.movie-score {
    background: #4c1d95;
    color: #c084fc;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 700;
    white-space: nowrap;
    margin-left: 12px;
}
.header-bar {
    background: #12122a;
    border-bottom: 1px solid #312e81;
    padding: 10px 0 18px;
    margin-bottom: 24px;
}
.tag {
    display: inline-block;
    background: #1e1e30;
    border: 1px solid #4c1d95;
    color: #a78bfa;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 11px;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="header-bar">', unsafe_allow_html=True)
st.title(" Movie Recommendation")
st.markdown('<span class="tag">Collaborative Filtering</span><span class="tag">Content-Based</span><span class="tag">Hybrid</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

def clean_genres(text):
    text = str(text).lower().replace('|', ' ')
    text = re.sub(r'[^a-z ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

@st.cache_resource(show_spinner=" Training model…")
def setup():
    movies  = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    if 'timestamp' in ratings.columns:
        ratings.drop(columns=['timestamp'], inplace=True)
    movies['genres'] = movies['genres'].apply(clean_genres)

    popular = ratings.groupby('movieId')['rating'].count()
    popular = popular[popular >= 20].index
    ratings = ratings[ratings['movieId'].isin(popular)]
    movies  = movies[movies['movieId'].isin(popular)]

    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    all_users  = sorted(ratings['userId'].unique())
    all_movies = sorted(ratings['movieId'].unique())
    user_idx   = {u: i for i, u in enumerate(all_users)}
    movie_idx  = {m: i for i, m in enumerate(all_movies)}

    R = np.zeros((len(all_users), len(all_movies)))
    for row in train_df.itertuples():
        R[user_idx[row.userId], movie_idx[row.movieId]] = row.rating

    user_mean  = np.true_divide(R.sum(1), (R != 0).sum(1).clip(1))
    R_centered = R.copy()
    for i in range(R.shape[0]):
        mask = R[i] != 0
        R_centered[i, mask] -= user_mean[i]

    k = min(50, min(R.shape) - 1)
    U, sigma, Vt = svds(R_centered, k=k)
    pred_matrix  = user_mean[:, np.newaxis] + U @ np.diag(sigma) @ Vt

    actual, estimated = [], []
    for row in test_df.itertuples():
        if row.userId in user_idx and row.movieId in movie_idx:
            p = float(np.clip(pred_matrix[user_idx[row.userId], movie_idx[row.movieId]], 1, 5))
            actual.append(row.rating); estimated.append(p)

    rmse = float(np.sqrt(mean_squared_error(actual, estimated)))
    mae  = float(mean_absolute_error(actual, estimated))

    mc      = movies[['movieId', 'title', 'genres']].drop_duplicates().reset_index(drop=True)
    tfidf   = TfidfVectorizer(stop_words='english').fit_transform(mc['genres'].fillna(''))
    cos_sim = cosine_similarity(tfidf, tfidf)
    idx_map = pd.Series(mc.index, index=mc['title']).drop_duplicates()

    return movies, ratings, pred_matrix, user_idx, movie_idx, mc, cos_sim, idx_map, rmse, mae

movies, ratings, pred_matrix, user_idx, movie_idx, mc, cos_sim, idx_map, rmse, mae = setup()

def hybrid(user_id, title, n=10, cf_w=0.6, cb_w=0.4):
    watched     = set(ratings[ratings['userId'] == user_id]['movieId'].values)
    not_watched = [m for m in movie_idx if m not in watched]

    if user_id in user_idx:
        ui = user_idx[user_id]
        cf = {m: float(np.clip(pred_matrix[ui, movie_idx[m]], 1, 5))
              for m in not_watched if m in movie_idx}
    else:
        g  = ratings['rating'].mean()
        cf = {m: g for m in not_watched}

    if not cf:
        return pd.DataFrame()

    lo, hi  = min(cf.values()), max(cf.values())
    cf_norm = {m: (s - lo) / (hi - lo + 1e-9) for m, s in cf.items()}

    cb, cb_norm = {}, {}
    if title in idx_map:
        for i, score in enumerate(cos_sim[idx_map[title]]):
            mid = mc.iloc[i]['movieId']
            if mid in cf_norm:
                cb[mid] = float(score)
    if cb:
        lo2, hi2 = min(cb.values()), max(cb.values())
        cb_norm  = {m: (s - lo2) / (hi2 - lo2 + 1e-9) for m, s in cb.items()}

    scores = {m: cf_w * cf_norm[m] + cb_w * cb_norm.get(m, 0) for m in cf_norm}
    top    = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    rows, count = [], 0
    for mid, score in top:
        if count == n: break
        row = movies[movies['movieId'] == mid]
        if row.empty: continue
        t = row['title'].values[0]
        if t == title: continue
        rows.append({'Title': t, 'Genres': row['genres'].values[0], 'Score': round(score, 3)})
        count += 1
    return pd.DataFrame(rows)

left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown(" Settings")
    user_id     = st.number_input("User ID", min_value=1, max_value=943, value=1)
    movie_title = st.selectbox("Movie you like", sorted(idx_map.index.tolist()))
    n_recs      = st.slider("Recommendations", 5, 20, 10)
    go          = st.button("Find Movies ")

with right:
    tab1, tab2 = st.tabs([" Recommendations", " Evaluation"])

    with tab1:
        if go:
            with st.spinner("Scanning the database…"):
                df = hybrid(int(user_id), movie_title, n_recs)
            if df.empty:
                st.warning("No recommendations found.")
            else:
                st.markdown(f"**{len(df)} movies for User {user_id}** · based on *{movie_title}*")
                st.write("")
                for i, row in df.iterrows():
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-info">
                            <div class="movie-num">#{i+1}</div>
                            <div class="movie-title">{row['Title']}</div>
                            <div class="movie-genre">{row['Genres']}</div>
                        </div>
                        <div class="movie-score">{row['Score']}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4b5563; margin-top:40px; text-align:center;">← Configure and click Find Movies</p>', unsafe_allow_html=True)

    with tab2:
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error on test set")
        c2.metric("MAE",  f"{mae:.4f}",  help="Mean Absolute Error on test set")
        st.markdown('<p style="color:#6b7280; font-size:12px; margin-top:8px;">Evaluated on 20% hold-out · SVD k=50 · CF 0.6 / CB 0.4</p>', unsafe_allow_html=True)