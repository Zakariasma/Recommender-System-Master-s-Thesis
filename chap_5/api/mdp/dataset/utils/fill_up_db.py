import os
import shutil
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, inspect

from chap_5.api.mdp.config import DATABASE_URL


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
    _seed(engine, movies, "movies")
    _seed(engine, genres, "genres")
    _seed(engine, movie_genre, "movie_genre")


def insert(engine, df, table_name):
    if df.empty:
        return

    if not inspect(engine).has_table(table_name):
        df.head(0).to_sql(table_name, engine, if_exists='append', index=False)

    cols = ",".join(f'"{c}"' for c in df.columns)
    query = f"INSERT INTO {table_name} ({cols}) VALUES %s"
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