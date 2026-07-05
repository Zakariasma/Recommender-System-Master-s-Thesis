import os
from dotenv import load_dotenv

from chap_5.api.mdp.config import RAW_DIR

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

DATASETS = {
    'historique': 'smaalizakaria/historique-thesis',
    'movies': 'smaalizakaria/movies',
    'genres': 'smaalizakaria/genres',
    'movie_genre': 'smaalizakaria/movie-genre',
    'movielens': 'grouplens/movielens-20m-dataset',
}


class DatasetRetriever:
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_API_TOKEN'] = os.getenv('KAGGLE_API_TOKEN')
        self.api = KaggleApi()
        self.api.authenticate()

    def _download(self, folder: str, dataset: str):
        path = RAW_DIR / folder
        path.mkdir(parents=True, exist_ok=True)
        if any(f.endswith('.csv') for f in os.listdir(path)):
            return
        self.api.dataset_download_files(dataset, path=str(path), unzip=True)

    def _find_csv(self, folder: str, name_hint: str = None) -> str:
        path = RAW_DIR / folder
        csvs = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csvs:
            raise FileNotFoundError(f"Aucun CSV trouvé dans {path}")
        if name_hint:
            match = [f for f in csvs if name_hint.lower() in f.lower()]
            if match:
                return str(path / match[0])
        return str(path / csvs[0])

    def download_all(self):
        for folder, dataset in DATASETS.items():
            self._download(folder, dataset)

    def load_data(self):
        self.download_all()
        historique = pd.read_csv(self._find_csv('historique'))
        movies = pd.read_csv(self._find_csv('movies'))
        genres = pd.read_csv(self._find_csv('genres'))
        movie_genre = pd.read_csv(self._find_csv('movie_genre'))
        ratings = pd.read_csv(self._find_csv('movielens', 'rating'), dtype={'userId': int, 'movieId': int})
        ml_movies = pd.read_csv(self._find_csv('movielens', 'movie'))
        links = pd.read_csv(self._find_csv('movielens', 'link'),
                             dtype={'movieId': int, 'imdbId': str, 'tmdbId': str})
        return historique, movies, genres, movie_genre, ratings, ml_movies, links
