from sqlalchemy import text

from chap_5.api.mdp.config import BOOTSTRAP, RAW_DIR
from chap_5.api.mdp.dataset.format_to_npy import build_sequences
from chap_5.api.mdp.dataset.init_genre_cache import init_genre_cache
from chap_5.api.mdp.dataset.normalize_data import preprocess
from chap_5.api.mdp.dataset.retrieve_data import DatasetRetriever
from chap_5.api.mdp.dataset.seed_database import (
    create_mdp_tables,
    drop_data_folder,
    get_engine,
    insert_csv_in_chunks,
    seed,
    table_exists,
)


def init_dataset():
    engine = get_engine()

    if table_exists(engine, "movies") and table_exists(engine, "genres"):
        if BOOTSTRAP:
            bootstrap_mdp_data(engine)
        return

    retriever = DatasetRetriever()
    historique, movies, genres, movie_genre, ratings, ml_movies, links = (
        retriever.load_data()
    )

    if not BOOTSTRAP:
        preprocess(historique, ratings, links, movies)
        build_sequences()

    seed(movies, genres, movie_genre)
    init_genre_cache()

    if BOOTSTRAP:
        bootstrap_mdp_data(engine)

    drop_data_folder(str(RAW_DIR.parent))


def bootstrap_mdp_data(engine):
    create_mdp_tables(engine)

    with engine.connect() as conn:
        trans_count = conn.execute(
            text("SELECT COUNT(*) FROM transitions")
        ).scalar()

        kv_count = conn.execute(
            text("SELECT COUNT(*) FROM kv_store")
        ).scalar()

    if trans_count > 0 and kv_count > 0:
        return

    retriever = DatasetRetriever()
    trans_path, kv_path = retriever.get_mdp_csv_paths()

    insert_csv_in_chunks(
        engine,
        trans_path,
        "transitions",
        chunksize=5000,
    )

    insert_csv_in_chunks(
        engine,
        kv_path,
        "kv_store",
        chunksize=5000,
    )


if __name__ == "__main__":
    init_dataset()
