import os
import shutil
import pandas as pd
from sqlalchemy import create_engine, text

from chap_5.api.mdp.config import DATABASE_URL


def get_engine():
    return create_engine(DATABASE_URL)


def table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :t)"
        ), {"t": table_name})
        return result.scalar()


def _seed_table(engine, df: pd.DataFrame, table_name: str):
    if table_exists(engine, table_name):
        return
    df.to_sql(table_name, engine, if_exists='replace', index=False)


def drop_data_folder(base_dir: str, keep: set = frozenset({'cleaned', 'npy'})):
    for folder in os.listdir(base_dir):
        if folder not in keep:
            shutil.rmtree(os.path.join(base_dir, folder))

def seed(movies: pd.DataFrame, genres: pd.DataFrame, movie_genre: pd.DataFrame):
    print("Connexion à la BD")
    engine = get_engine()
    print("Insert movies")
    _seed_table(engine, movies, 'movies')
    print("Insert genres")
    _seed_table(engine, genres, 'genres')
    print("Insert movie_genre")
    _seed_table(engine, movie_genre, 'movie_genre')
