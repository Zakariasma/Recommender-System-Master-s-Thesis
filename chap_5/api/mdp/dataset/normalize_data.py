import pandas as pd

from chap_5.api.mdp.config import CLEANED_DIR


def clean_historique(historique: pd.DataFrame) -> pd.DataFrame:
    return historique[historique['media_type'] != 'series'].reset_index(drop=True)


def build_movielens_historique(ratings: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    links_clean = links.copy()
    links_clean['imdbId'] = links_clean['imdbId'].apply(
        lambda x: f"tt{str(x).zfill(7)}" if pd.notna(x) else None
    )
    links_clean = links_clean.rename(columns={'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'})

    ratings_filtered = ratings[ratings['rating'] >= 4.0]
    merged = ratings_filtered.merge(links_clean[['movieId', 'imdb_id', 'tmdb_id']], on='movieId', how='left')
    merged = merged[['userId', 'movieId', 'tmdb_id', 'imdb_id', 'timestamp']]
    return merged.sort_values(['userId', 'timestamp']).reset_index(drop=True)


def save_cleaned(df: pd.DataFrame, filename: str):
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    path = CLEANED_DIR / filename
    df.to_csv(path, index=False)
    print(f"  -> {path}")


def _match_by_id(source: pd.DataFrame, movies: pd.DataFrame, id_cols: list, keep_cols: list) -> pd.DataFrame:
    movies_clean = movies.copy().rename(columns={'id': 'movie_id'})
    movies_clean['tmdb_id'] = movies_clean['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    movies_clean['imdb_id'] = movies_clean['imdb_id'].astype(str).str.strip()

    src = source.copy()
    src['_row_id'] = range(len(src))
    src['tmdb_id'] = src['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    src['imdb_id'] = src['imdb_id'].astype(str).str.strip()

    by_tmdb = src.merge(movies_clean[['movie_id', 'tmdb_id']], on='tmdb_id', how='inner')
    remaining = src.loc[~src['_row_id'].isin(by_tmdb['_row_id'])]
    by_imdb = remaining.merge(movies_clean[['movie_id', 'imdb_id']], on='imdb_id', how='inner')

    matched = pd.concat([by_tmdb[keep_cols], by_imdb[keep_cols]], ignore_index=True)
    return matched.drop_duplicates(subset=id_cols)


def match_movielens_to_movies(ml_hist: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    cols = ['userId', 'movie_id', 'timestamp']
    matched = _match_by_id(ml_hist, movies, cols, cols)
    print(f"  movielens match : {len(matched)} lignes, {matched['userId'].nunique()} utilisateurs")
    return matched


def match_historique_to_movies(hist_clean: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    cols = ['id_address', 'movie_id', 'created_at']
    matched = _match_by_id(hist_clean, movies, cols, cols)
    result = matched.rename(columns={'id_address': 'ip', 'created_at': 'timestamp'})
    print(f"  historique match : {len(result)} lignes, {result['ip'].nunique()} utilisateurs")
    return result


def merge_historiques(ml_matched: pd.DataFrame, hist_matched: pd.DataFrame) -> pd.DataFrame:
    max_user_id = int(ml_matched['userId'].max())
    ip_to_id = {ip: max_user_id + i + 1 for i, ip in enumerate(hist_matched['ip'].unique())}

    hist = hist_matched.copy()
    hist['userId'] = hist['ip'].map(ip_to_id)
    hist = hist[['userId', 'movie_id', 'timestamp']]
    hist['timestamp'] = pd.to_datetime(hist['timestamp'], utc=True)

    ml = ml_matched[['userId', 'movie_id', 'timestamp']].copy()
    ml['timestamp'] = pd.to_datetime(ml['timestamp'], unit='s', utc=True)

    merged = pd.concat([ml, hist], ignore_index=True).sort_values(['userId', 'timestamp']).reset_index(drop=True)
    print(f"  full_hist : {len(merged)} lignes, {merged['userId'].nunique()} utilisateurs")
    return merged


def preprocess(historique: pd.DataFrame, ratings: pd.DataFrame, links: pd.DataFrame, movies: pd.DataFrame):
    print("\n[preprocess]")
    hist_clean = clean_historique(historique)
    save_cleaned(hist_clean, 'historique_clean.csv')

    ml_hist = build_movielens_historique(ratings, links)
    save_cleaned(ml_hist, 'movielens_historique.csv')

    ml_matched = match_movielens_to_movies(ml_hist, movies)
    hist_matched = match_historique_to_movies(hist_clean, movies)

    full_hist = merge_historiques(ml_matched, hist_matched)
    save_cleaned(full_hist, 'historique_full.csv')

    return hist_clean, ml_hist, full_hist
