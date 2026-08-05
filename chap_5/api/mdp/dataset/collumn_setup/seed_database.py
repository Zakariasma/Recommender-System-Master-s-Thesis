import os
import shutil

import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    Table,
    create_engine,
    func,
    text,
)

from chap_5.api.mdp.config import DATABASE_URL
from chap_5.api.mdp.dataset.collumn_setup.setup_mdp_tables import create_mdp_tables


metadata = MetaData()

movie_history = Table(
    "movie_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("movie_id", Integer, nullable=False),
    Column("viewed_at", DateTime(timezone=True), server_default=func.now()),
)


def get_engine():
    return create_engine(DATABASE_URL)


def table_exists(engine, table_name: str) -> bool:
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = :t
                )
                """
            ),
            {"t": table_name},
        )
        return result.scalar()


def _seed_table(engine, df: pd.DataFrame, table_name: str):
    if table_exists(engine, table_name):
        return

    df.to_sql(table_name, engine, if_exists="replace", index=False)


def drop_data_folder(
    base_dir: str,
    keep: set = frozenset({"cleaned", "npy"}),
):
    for folder in os.listdir(base_dir):
        if folder not in keep:
            shutil.rmtree(os.path.join(base_dir, folder))


def seed(
    movies: pd.DataFrame,
    genres: pd.DataFrame,
    movie_genre: pd.DataFrame,
):
    engine = get_engine()

    metadata.create_all(engine, tables=[movie_history])

    create_mdp_tables(engine)

    _seed_table(engine, movies, "movies")
    _seed_table(engine, genres, "genres")
    _seed_table(engine, movie_genre, "movie_genre")


def fast_insert(engine, df: pd.DataFrame, table_name: str):
    if df.empty:
        return

    tuples = [tuple(row) for row in df.to_numpy()]
    cols = ",".join(f'"{col}"' for col in df.columns)
    query = f"INSERT INTO {table_name} ({cols}) VALUES %s"

    raw_conn = engine.raw_connection()

    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, query, tuples, page_size=10000)

        raw_conn.commit()

    except Exception:
        raw_conn.rollback()
        raise

    finally:
        raw_conn.close()


def insert_csv_in_chunks(
    engine,
    csv_path: str,
    table_name: str,
    chunksize: int = 50000,
):
    dtype = None

    if table_name == "transitions":
        dtype = {"s": str, "s_": str}
    elif table_name == "kv_store":
        dtype = {"namespace": str, "key": str, "value": str}

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=dtype,
    ):
        fast_insert(engine, chunk, table_name)