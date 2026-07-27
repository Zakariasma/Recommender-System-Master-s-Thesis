from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfMatrixBuilder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def build(self, movies_df):
        movies_df['genres_clean'] = movies_df['genres'].str.replace('|', ' ', regex=False)
        tfidf_matrix = self.vectorizer.fit_transform(movies_df['genres_clean'])
        return tfidf_matrix