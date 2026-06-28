import os
import sys
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def get_engine():
    url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/master')
    return create_engine(url)


def table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :t)"
        ), {"t": table_name})
        return result.scalar()


def seed_movies(engine, movies: pd.DataFrame):
    if table_exists(engine, 'movies'):
        return
    movies.to_sql('movies', engine, if_exists='replace', index=False)


def seed_genres(engine, genres: pd.DataFrame):
    if table_exists(engine, 'genres'):
        return
    genres.to_sql('genres', engine, if_exists='replace', index=False)


def seed_movie_genre(engine, movie_genre: pd.DataFrame):
    if table_exists(engine, 'movie_genre'):
        return
    movie_genre.to_sql('movie_genre', engine, if_exists='replace', index=False)


def seed_historique(engine, full_hist: pd.DataFrame):
    if table_exists(engine, 'historique'):
        return

    batch_size = 10_000
    total = len(full_hist)
    batches = (total // batch_size) + (1 if total % batch_size else 0)

    for i in range(batches):
        batch = full_hist.iloc[i * batch_size:(i + 1) * batch_size]
        batch.to_sql(
            'historique', engine,
            if_exists='replace' if i == 0 else 'append',
            index=False
        )


def drop_data_folder(base_dir: str):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)


def seed(movies: pd.DataFrame, genres: pd.DataFrame, movie_genre: pd.DataFrame, full_hist: pd.DataFrame):
    print("Connexion à la BD")
    engine = get_engine()
    print("Insert movies")
    seed_movies(engine, movies)
    print("Insert genres")
    seed_genres(engine, genres)
    seed_movie_genre(engine, movie_genre)
    print("Insert historiques")
    seed_historique(engine, full_hist)