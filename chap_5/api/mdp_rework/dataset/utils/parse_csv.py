import pandas as pd
from chap_5.api.mdp.config import CLEANED_DIR


def _save(df, name):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DIR / name, index=False)


def _match(source, movies, u_col, t_col):
    m = movies[['id', 'tmdb_id', 'imdb_id']].copy()
    m['tmdb_id'] = m['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    m['imdb_id'] = m['imdb_id'].astype(str).str.strip()

    s = source[[u_col, t_col, 'tmdb_id', 'imdb_id']].copy()
    s['tmdb_id'] = s['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    s['imdb_id'] = s['imdb_id'].astype(str).str.strip()

    by_tmdb = s.merge(m[['id', 'tmdb_id']], on='tmdb_id')
    by_imdb = s[~s.index.isin(by_tmdb.index)].merge(m[['id', 'imdb_id']], on='imdb_id')

    res = pd.concat([by_tmdb, by_imdb]).rename(columns={'id': 'movie_id', u_col: 'user_id', t_col: 'timestamp'})
    return res[['user_id', 'movie_id', 'timestamp']].drop_duplicates(subset=['user_id', 'movie_id'])


def parse_csv(historique, ratings, links, movies):
    hist_clean = historique[historique['media_type'] != 'series'].reset_index(drop=True)
    _save(hist_clean, 'historique_clean.csv')

    ml = ratings[ratings['rating'] >= 4.0][['userId', 'movieId', 'timestamp']].merge(
        links[['movieId', 'imdbId', 'tmdbId']], on='movieId'
    )
    ml['imdb_id'] = 'tt' + ml['imdbId'].astype(str).str.zfill(7)
    ml['tmdb_id'] = ml['tmdbId'].astype(str).str.replace('.0', '', regex=False)
    _save(ml[['userId', 'movieId', 'timestamp', 'imdb_id', 'tmdb_id']], 'movielens_historique.csv')

    ml_matched = _match(ml, movies, 'userId', 'timestamp')
    hist_matched = _match(hist_clean, movies, 'id_address', 'created_at')

    max_id = int(ml_matched['user_id'].max())
    ip_map = {ip: max_id + i + 1 for i, ip in enumerate(hist_matched['user_id'].unique())}
    hist_matched['user_id'] = hist_matched['user_id'].map(ip_map)

    ml_matched['timestamp'] = pd.to_datetime(ml_matched['timestamp'], unit='s', utc=True)
    hist_matched['timestamp'] = pd.to_datetime(hist_matched['timestamp'], utc=True)

    full = pd.concat([ml_matched, hist_matched], ignore_index=True).sort_values(['user_id', 'timestamp']).reset_index(
        drop=True)
    _save(full, 'historique_full.csv')
    return full