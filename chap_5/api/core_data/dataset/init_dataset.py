import os
import sys

from chap_5.api.core_data.dataset.normalize_data import preprocess
from chap_5.api.core_data.dataset.seed_database import seed, drop_data_folder, get_engine, table_exists
from chap_5.api.core_data.dataset.retrieve_data import DatasetRetriever, BASE_DIR

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


FULL_HIST_PATH = os.path.join(os.path.dirname(__file__), 'core_data', 'cleaned', 'full_hist.csv')

def init_dataset():
    engine = get_engine()
    if table_exists(engine, 'movies') and table_exists(engine, 'genres'):
        print("Tables déjà existantes, skip.")
        return

    print("Téléchargement des datasets...")
    retriever = DatasetRetriever()
    retriever.download_all()

    print("Chargement des données...")
    historique, movies, genres, movie_genre, ratings, ml_movies, links = retriever.load_data()

    print("Prétraitement...")
    hist_clean, ml_hist, full_hist = preprocess(historique, ratings, links, movies)

    print("Seed en bd...")
    seed(movies, genres, movie_genre)

    print("Nettoyage...")
    drop_data_folder(BASE_DIR)

    print("Données initialisées.")


if __name__ == "__main__":
    init_dataset()