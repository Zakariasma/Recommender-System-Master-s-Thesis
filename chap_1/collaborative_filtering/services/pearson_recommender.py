from surprise import KNNBasic
from surprise import Dataset, Reader

class PearsonRecommender:
    def __init__(self, ratings_df, movies_df, k=50):
        self.movies_df = movies_df
        self.k = k

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()

        sim_options = {'name': 'pearson', 'user_based': True}

        self.algo = KNNBasic(k=self.k, sim_options=sim_options, verbose=False)
        self.algo.fit(trainset)
        self.trainset = trainset

    def recommend(self, user_id, top_k=10):
        # Traduire id user en id interne
        inner_uid = self.trainset.to_inner_uid(user_id)

        # Recup liste des films déjà vus par l'user
        seen = [iid for (iid, _) in self.trainset.ur[inner_uid]]
        predictions = []

        # Tout les films du trainset
        for inner_iid in self.trainset.all_items():

            # Skip films deja vu par user
            if inner_iid in seen:
                continue

            # Traduire id interne en id movieLens
            raw_iid = self.trainset.to_raw_iid(inner_iid)

            # Prediction note
            pred = self.algo.predict(user_id, raw_iid, verbose=False)
            predictions.append((raw_iid, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)

        # Garde top-k et recep titres réels des films
        results = []
        for movie_id, score in predictions[:top_k]:
            title = self.movies_df[self.movies_df['movieId'] == int(movie_id)]['title'].values[0]
            results.append((title, round(score, 4)))

        return results