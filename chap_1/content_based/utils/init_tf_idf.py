import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

load_dotenv()


class InitMovieTfIdf:
    def __init__(self, movies_path='../../shared/data/movies/movies_dataset_kaggle.csv'):
        self.movies_df = pd.read_csv(movies_path)
        self.vectors_path = 'data/movie_tf_idf_vectors.npy'
        os.makedirs('data', exist_ok=True)

    def preprocess_text(self, row):
        def safe_join(col):
            if pd.isna(col):
                return ''
            if isinstance(col, list):
                return ' '.join([str(x) for x in col if x])
            return str(col)

        plot = safe_join(row.get('plot', ''))
        genres = safe_join(row.get('genres', ''))
        themes = safe_join(row.get('themes', ''))

        text = f"{plot} {genres} {themes}".strip()
        return text if text else 'unknown'

    def create_vectors(self):
        if os.path.exists(self.vectors_path):
            return np.load(self.vectors_path)

        self.movies_df['text'] = self.movies_df.apply(self.preprocess_text, axis=1)
        valid_mask = self.movies_df['text'].str.len() > 0
        self.movies_df = self.movies_df[valid_mask].reset_index(drop=True)
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )

        tfidf_matrix = vectorizer.fit_transform(self.movies_df['text'])

        # Besoin convertir sinon pb pdt ouverture
        vect = normalize(tfidf_matrix, norm='l2').toarray().astype(np.float32)
        np.save(self.vectors_path, vect)
        #self.print_sample(vectorizer, vect)
        return vect

    def print_sample(self, vectorizer, vectors, n_samples=3):
        feature_names = np.array(vectorizer.get_feature_names_out())
        for i in range(min(n_samples, vectors.shape[0])):
            title = self.movies_df.iloc[i]['title'][:40] + "..."
            row = vectors[i]

            top5_idx = np.argsort(row)[-5:][::-1]
            top5_words = feature_names[top5_idx]
            top5_weights = row[top5_idx]

            print(f"Film {i}: {title}")
            print(f"  Top 5: {list(zip(top5_words, np.round(top5_weights, 4)))}")
            print(f"  Non-zéros: {len(vectors[i].data)}, somme: {vectors[i].sum():.4f}\n")

    def get_vectors(self):
        return self.create_vectors()


if __name__ == "__main__":
    tfidf = InitMovieTfIdf()
    vectors = tfidf.get_vectors()
