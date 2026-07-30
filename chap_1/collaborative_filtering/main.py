from chap_1.collaborative_filtering.services.pearson_recommender import PearsonRecommender
from chap_1.shared.init_dataset import DatasetRetriever

USER_ID = 10
TOP_K = 15

if __name__ == "__main__":
    retriever = DatasetRetriever()
    ratings, movies = retriever.load_data()

    # On garde les 1k users avec plus de note, evite faire exploser RAM/Combinaison à faire
    active_users = set(ratings['userId'].value_counts().head(1000).index)
    active_users.add(USER_ID)
    ratings_subset = ratings[ratings['userId'].isin(active_users)]

    print(f"Pearson Recommendations (Notes) for User {USER_ID} ---")
    pearson_rec = PearsonRecommender(ratings_subset, movies)
    for title, score in pearson_rec.recommend(USER_ID, TOP_K):
        print(f"- {title} (Score: {score})")
