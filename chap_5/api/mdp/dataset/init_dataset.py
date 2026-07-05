from chap_5.api.mdp.dataset.format_to_npy import build_sequences
from chap_5.api.mdp.dataset.normalize_data import preprocess
from chap_5.api.mdp.dataset.seed_database import seed, drop_data_folder, get_engine, table_exists
from chap_5.api.mdp.dataset.retrieve_data import DatasetRetriever
from chap_5.api.mdp.config import RAW_DIR


def init_dataset():
    engine = get_engine()
    if table_exists(engine, 'movies') and table_exists(engine, 'genres'):
        print("Tables déjà existantes, skip.")
        return

    print("Téléchargement des datasets...")
    retriever = DatasetRetriever()
    historique, movies, genres, movie_genre, ratings, ml_movies, links = retriever.load_data()

    print("Prétraitement...")
    preprocess(historique, ratings, links, movies)
    build_sequences()

    print("Seed en bd...")
    seed(movies, genres, movie_genre)

    print("Nettoyage...")
    drop_data_folder(str(RAW_DIR.parent))

    print("Données initialisées.")


if __name__ == "__main__":
    init_dataset()
