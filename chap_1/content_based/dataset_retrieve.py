import os
from dotenv import load_dotenv

load_dotenv()

# Vérification token kaggle auth
username = os.getenv('KAGGLE_USERNAME')
token = os.getenv('KAGGLE_API_TOKEN')

os.environ['KAGGLE_USERNAME'] = username
os.environ['KAGGLE_API_TOKEN'] = token

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

api = KaggleApi()
api.authenticate()

os.makedirs('./data', exist_ok=True)

api.dataset_download_files('smaalizakaria/umons-smaali-cs-movie-one-thesis',
                          path='./data/movies/', unzip=True)

api.dataset_download_files('netflix-inc/netflix-prize-data',
                          path='./data/netflix/', unzip=True)

movies_df = pd.read_csv('./data/movies/movies_dataset_kaggle.csv')
netflix_files = [f for f in os.listdir('./data/netflix/') if f.startswith('combined_data_')]

print(f"Movies: {len(movies_df)} films")
print(f"Netflix: {len(netflix_files)} fichiers")

