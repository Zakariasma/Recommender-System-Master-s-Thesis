import os
from dotenv import load_dotenv

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


class DatasetRetriever:
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_API_TOKEN'] = os.getenv('KAGGLE_API_TOKEN')
        self.api = KaggleApi()
        self.api.authenticate()

    def download_movies(self):
        if not os.path.exists(os.path.join(BASE_DIR, 'movies/movies_dataset_kaggle.csv')):
            self.api.dataset_download_files('smaalizakaria/umons-smaali-cs-movie-one-thesis',
                                            path=os.path.join(BASE_DIR, 'movies'), unzip=True)

    def download_movielens(self):
        if not os.path.exists(os.path.join(BASE_DIR, 'movielens/rating.csv')):
            self.api.dataset_download_files('grouplens/movielens-20m-dataset',
                                            path=os.path.join(BASE_DIR, 'movielens'), unzip=True)

    def load_data(self):
        self.download_movies()
        self.download_movielens()

        movies = pd.read_csv(os.path.join(BASE_DIR, 'movies/movies_dataset_kaggle.csv'))
        ratings = pd.read_csv(os.path.join(BASE_DIR, 'movielens/rating.csv'), dtype={'userId': int, 'movieId': int})
        ml_movies = pd.read_csv(os.path.join(BASE_DIR, 'movielens/movie.csv'))
        links = pd.read_csv(os.path.join(BASE_DIR, 'movielens/link.csv'), dtype={'movieId': int, 'imdbId': str, 'tmdbId': str})

        return movies, ratings, ml_movies, links


if __name__ == "__main__":
    retriever = DatasetRetriever()
    movies, ratings, ml_movies, links = retriever.load_data()
