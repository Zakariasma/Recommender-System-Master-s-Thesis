import os
import sys

from chap_5.api.core_data.dataset.retrieve_data import BASE_DIR

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd


def clean_historique(historique: pd.DataFrame) -> pd.DataFrame:
    before = len(historique)
    historique = historique[historique['media_type'] != 'series'].reset_index(drop=True)
    print(f"  historique : {before} → {len(historique)} lignes (series retirées)")
    return historique


def build_movielens_historique(ratings: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    links_clean = links.copy()
    links_clean['imdbId'] = links_clean['imdbId'].apply(
        lambda x: f"tt{str(x).zfill(7)}" if pd.notna(x) else None
    )
    links_clean = links_clean.rename(columns={'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'})

    ratings_filtered = ratings[ratings['rating'] >= 4.0]

    merged = ratings_filtered.merge(links_clean[['movieId', 'imdb_id', 'tmdb_id']], on='movieId', how='left')
    merged = merged[['userId', 'movieId', 'tmdb_id', 'imdb_id', 'timestamp']]
    merged = merged.sort_values(['userId', 'timestamp']).reset_index(drop=True)

    print(f"  movielens : {len(merged)} lignes, {merged['userId'].nunique()} utilisateurs")
    return merged


def save_cleaned(df: pd.DataFrame, filename: str):
    out_dir = os.path.join(BASE_DIR, 'cleaned')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"  -> {path}")


def match_movielens_to_movies(ml_hist: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    movies_clean = movies.copy().rename(columns={'id': 'movie_id'})
    movies_clean['tmdb_id'] = movies_clean['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    movies_clean['imdb_id'] = movies_clean['imdb_id'].astype(str).str.strip()

    ml = ml_hist.copy()
    ml['_row_id'] = range(len(ml))
    ml['tmdb_id'] = ml['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    ml['imdb_id'] = ml['imdb_id'].astype(str).str.strip()

    by_tmdb = ml.merge(
        movies_clean[['movie_id', 'tmdb_id']],
        on='tmdb_id',
        how='inner'
    )

    matched_row_ids = set(by_tmdb['_row_id'])
    remaining = ml.loc[~ml['_row_id'].isin(matched_row_ids)].copy()

    by_imdb = remaining.merge(
        movies_clean[['movie_id', 'imdb_id']],
        on='imdb_id',
        how='inner'
    )

    matched = pd.concat([by_tmdb[['userId', 'movie_id', 'timestamp']],
                         by_imdb[['userId', 'movie_id', 'timestamp']]],
                        ignore_index=True)
    matched = matched.drop_duplicates(subset=['userId', 'movie_id', 'timestamp'])

    print(f"  movielens match : {len(matched)} lignes, {matched['userId'].nunique()} utilisateurs")
    return matched


def match_historique_to_movies(hist_clean: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    movies_clean = movies.copy().rename(columns={'id': 'movie_id'})
    movies_clean['tmdb_id'] = movies_clean['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    movies_clean['imdb_id'] = movies_clean['imdb_id'].astype(str).str.strip()

    hist = hist_clean.copy()
    hist['_row_id'] = range(len(hist))
    hist['tmdb_id'] = hist['tmdb_id'].astype(str).str.strip().str.replace('.0', '', regex=False)
    hist['imdb_id'] = hist['imdb_id'].astype(str).str.strip()

    by_tmdb = hist.merge(
        movies_clean[['movie_id', 'tmdb_id']],
        on='tmdb_id',
        how='inner'
    )

    matched_row_ids = set(by_tmdb['_row_id'])
    remaining = hist.loc[~hist['_row_id'].isin(matched_row_ids)].copy()

    by_imdb = remaining.merge(
        movies_clean[['movie_id', 'imdb_id']],
        on='imdb_id',
        how='inner'
    )

    matched = pd.concat(
        [by_tmdb[['id_address', 'movie_id', 'created_at']],
         by_imdb[['id_address', 'movie_id', 'created_at']]],
        ignore_index=True
    )
    matched = matched.drop_duplicates(subset=['id_address', 'movie_id', 'created_at'])

    result = matched.rename(columns={'id_address': 'ip', 'created_at': 'timestamp'})
    print(f"  historique match : {len(result)} lignes, {result['ip'].nunique()} utilisateurs")
    return result


def merge_historiques(ml_matched: pd.DataFrame, hist_matched: pd.DataFrame) -> pd.DataFrame:
    max_user_id = int(ml_matched['userId'].max())
    ip_to_id = {ip: max_user_id + i + 1 for i, ip in enumerate(hist_matched['ip'].unique())}

    hist = hist_matched.copy()
    hist['userId'] = hist['ip'].map(ip_to_id)
    hist = hist[['userId', 'movie_id', 'timestamp']]

    ml = ml_matched[['userId', 'movie_id', 'timestamp']].copy()

    ml['timestamp'] = pd.to_datetime(ml['timestamp'], unit='s', utc=True)
    hist['timestamp'] = pd.to_datetime(hist['timestamp'], utc=True)

    merged = pd.concat([ml, hist], ignore_index=True)
    merged = merged.sort_values(['userId', 'timestamp']).reset_index(drop=True)

    print(f"  full_hist : {len(merged)} lignes, {merged['userId'].nunique()} utilisateurs")
    return merged


def preprocess(historique: pd.DataFrame,
               ratings: pd.DataFrame,
               links: pd.DataFrame,
               movies: pd.DataFrame):

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