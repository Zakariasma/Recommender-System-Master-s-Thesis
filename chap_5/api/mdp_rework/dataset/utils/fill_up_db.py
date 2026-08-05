import os
import shutil
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, DateTime, func, inspect

from chap_5.api.mdp.config import DATABASE_URL
from chap_5.api.mdp.dataset.collumn_setup.setup_mdp_tables import create_mdp_tables

metadata = MetaData()

movie_history = Table(
    "movie_history", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("movie_id", Integer, nullable=False),
    Column("viewed_at", DateTime(timezone=True), server_default=func.now()),
)


def get_engine():
    return create_engine(DATABASE_URL)


def _seed(engine, df, name):
    if not inspect(engine).has_table(name):
        df.to_sql(name, engine, if_exists="replace", index=False)


def drop_data_folder(base_dir, keep=frozenset({"cleaned", "npy"})):
    for f in os.listdir(base_dir):
        if f not in keep:
            shutil.rmtree(os.path.join(base_dir, f))


def seed(movies, genres, movie_genre):
    engine = get_engine()
    metadata.create_all(engine, tables=[movie_history])
    create_mdp_tables(engine)
    _seed(engine, movies, "movies")
    _seed(engine, genres, "genres")
    _seed(engine, movie_genre, "movie_genre")


def insert(engine, df, table_name):
    if df.empty:
        return

    # Crée la table si elle n'existe pas
    if not inspect(engine).has_table(table_name):
        df.head(0).to_sql(table_name, engine, if_exists='append', index=False)

    cols = ",".join(f'"{c}"' for c in df.columns)
    query = f"INSERT INTO {table_name} ({cols}) VALUES %s"

    # Conversion des lignes en tuples de types Python natifs
    records = df.itertuples(index=False, name=None)

    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            execute_values(cur, query, records, page_size=10000)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_csv_in_chunks(engine, csv_path, table_name, chunksize=50000):
    dtypes = {
        "transitions": {"s": str, "s_": str},
        "kv_store": {"namespace": str, "key": str, "value": str}
    }.get(table_name)

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=dtypes):
        insert(engine, chunk, table_name)