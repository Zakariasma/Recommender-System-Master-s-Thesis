from chap_5.api.mdp_rework.dataset.utils.fill_up_db import insert_csv_in_chunks, get_engine, insert
from chap_5.api.mdp_rework.dataset.utils.frontend_data_caching import init_frontend_cache
from chap_5.api.mdp_rework.dataset.utils.parse_csv import parse_csv
from chap_5.api.mdp_rework.dataset.utils.transform_to_npy import transform_to_npy
from chap_5.api.mdp_rework.dataset.retrieve_data import DatasetRetriever


def init():
    engine = get_engine()
    retriever = DatasetRetriever()
    historique, movies, genres, movie_genre, ratings, ml_movies, links = (
        retriever.load_data()
    )
    parse_csv(historique, ratings, links, movies)
    transform_to_npy()
    insert(engine, movies, "movies")
    insert(engine, genres, "genres")
    insert(engine, movie_genre, "movie_genre")
    init_frontend_cache()

if __name__ == "__main__":
    init()
