import os
import sys

from chap_5.api.data.normalize_data import preprocess
from chap_5.api.data.seed_database import seed, drop_data_folder
from chap_5.api.data.retrieve_data import DatasetRetriever, BASE_DIR, DATASETS

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def init():
    print("Téléchargement des datasets...")
    retriever = DatasetRetriever()
    retriever.download_all()

    print("Chargement des données...")
    data = retriever.load_data()
    historique, movies, genres, movie_genre, ratings, ml_movies, links = data

    print("Prétraitement...")
    hist_clean, ml_hist, full_hist = preprocess(historique, ratings, links, movies)

    print("Seed en bd...")
    seed(movies, genres, movie_genre, full_hist)

    print("Nettoyage...")
    drop_data_folder(BASE_DIR)

    print("Terminé.")
    return data


if __name__ == "__main__":
    init()