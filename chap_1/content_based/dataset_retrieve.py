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

    def _movies_exists(self):
        return os.path.exists('./data/movies/movies_dataset_kaggle.csv')

    def _netflix_exists(self):
        return os.path.exists('./data/netflix/') and len(os.listdir('./data/netflix/')) > 0

    def download_movies(self):
        if not self._movies_exists():
            self.api.dataset_download_files('smaalizakaria/umons-smaali-cs-movie-one-thesis',
                                            path='./data/movies/', unzip=True)

    def download_netflix(self):
        if not self._netflix_exists():
            self.api.dataset_download_files('netflix-inc/netflix-prize-data',
                                            path='./data/netflix/', unzip=True)

    def load_data(self):
        movies_df = pd.read_csv('./data/movies/movies_dataset_kaggle.csv')
        netflix_files = [f for f in os.listdir('./data/netflix/') if f.startswith('combined_data_')]
        print(f"Movies: {len(movies_df)} films")
        print(f"Netflix: {len(netflix_files)} fichiers")
        return movies_df, netflix_files

    def get_all(self):
        self.download_movies()
        self.download_netflix()
        return self.load_data()


if __name__ == "__main__":
    retriever = DatasetRetriever()
    movies, netflix_files = retriever.get_all()
