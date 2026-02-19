import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

load_dotenv()


class InitMovieTfIdf:
    def __init__(self, movies_path='./data/movies/movies_dataset_kaggle.csv'):
        self.movies_df = pd.read_csv(movies_path)
        self.vectors_path = './data/movie_tf_idf_vectors.npy'
        os.makedirs('./data', exist_ok=True)

    def preprocess_text(self, row):
        # Combine plot, genres, themes en un seul texte
        plot = row['plot'] if row['plot'] != 'N/A' else ''
        genres = ' '.join(row['genres']) if row['genres'] and isinstance(row['genres'], list) else ''
        themes = ' '.join(row['themes']) if row['themes'] and isinstance(row['themes'], list) else ''
        return f"{plot} {genres} {themes}".strip()

    def create_vectors(self):
        # Check si existe déjà
        if os.path.exists(self.vectors_path):
            print("Vectors déjà créés")
            return np.load(self.vectors_path)

        # Crée TF-IDF matrice
        self.movies_df['text'] = self.movies_df.apply(self.preprocess_text, axis=1)
        self.movies_df.dropna(subset=['text'], inplace=True)

        # TF-IDF avec 5000 mots max et bigramme cad themes en un mots ou deux :
        # ex: "action" ou "action thriller"
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(self.movies_df['text'])
        vectors = normalize(tfidf_matrix)

        np.save(self.vectors_path, vectors)
        print(f"Vectors sauvegardés: {vectors.shape}")
        return vectors

    def get_vectors(self):
        return self.create_vectors()


if __name__ == "__main__":
    tfidf = InitMovieTfIdf()
    vectors = tfidf.get_vectors()
