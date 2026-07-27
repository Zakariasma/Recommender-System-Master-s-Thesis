import os
import glob
from dotenv import load_dotenv

load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

class DatasetRetriever:
    def __init__(self):
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_API_TOKEN'] = os.getenv('KAGGLE_API_TOKEN')
        self.api = KaggleApi()
        self.api.authenticate()

    def clean_folder(self):
        for file in glob.glob(os.path.join(BASE_DIR, "*.csv")):
            if os.path.basename(file) not in ['rating.csv', 'movie.csv']:
                os.remove(file)

    def download_movielens(self):
        if not os.path.exists(os.path.join(BASE_DIR, 'movie.csv')):
            self.api.dataset_download_files('grouplens/movielens-20m-dataset',
                                            path=BASE_DIR, unzip=True)
            self.clean_folder()

    def load_data(self):
        self.download_movielens()
        ratings = pd.read_csv(os.path.join(BASE_DIR, 'rating.csv'), dtype={'userId': int, 'movieId': int})
        ml_movies = pd.read_csv(os.path.join(BASE_DIR, 'movie.csv'))
        return ratings, ml_movies


if __name__ == "__main__":
    retriever = DatasetRetriever()
    ratings, ml_movies = retriever.load_data()