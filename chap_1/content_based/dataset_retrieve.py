import os
from dotenv import load_dotenv

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd


class DatasetRetriever:
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_API_TOKEN'] = os.getenv('KAGGLE_API_TOKEN')
        self.api = KaggleApi()
        self.api.authenticate()
        os.makedirs('./data', exist_ok=True)

    def download_movies(self):
        if not os.path.exists('./data/movies/movies_dataset_kaggle.csv'):
            self.api.dataset_download_files('smaalizakaria/umons-smaali-cs-movie-one-thesis',
                                            path='./data/movies/', unzip=True)

    def download_movielens(self):
        if not os.path.exists('./data/movielens/ratings.csv') and not os.path.exists('./data/movielens/rating.csv'):
            self.api.dataset_download_files('grouplens/movielens-20m-dataset',
                                            path='./data/movielens/', unzip=True)

    def load_data(self):
        self.download_movies()
        self.download_movielens()

        movies = pd.read_csv('./data/movies/movies_dataset_kaggle.csv')
        ratings = pd.read_csv('./data/movielens/rating.csv', dtype={'userId': int, 'movieId': int})
        ml_movies = pd.read_csv('./data/movielens/movie.csv')
        links = pd.read_csv('./data/movielens/link.csv', dtype={'movieId': int, 'imdbId': str, 'tmdbId': str})

        return movies, ratings, ml_movies, links


if __name__ == "__main__":
    retriever = DatasetRetriever()
    movies, ratings, ml_movies, links = retriever.load_data()
