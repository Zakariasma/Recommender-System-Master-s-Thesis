from chap_1.shared.utils.dataset_retrieve import DatasetRetriever
from chap_1.content_based.content_based_recommander import ContentBasedRecommender
from chap_1.shared.utils.map_user_history import export_user_movie_ids_csv

USER_ID = 10
TOP_K = 20
USER_HISTORY_PATH = "data/user_history/user_movie_ids.csv"
MOVIES_PATH = "../shared/data/movies/movies_dataset_kaggle.csv"

if __name__ == "__main__":
    DatasetRetriever().load_data()
    export_user_movie_ids_csv(USER_HISTORY_PATH)
    rec = ContentBasedRecommender(user_history_path=USER_HISTORY_PATH, movies_path=MOVIES_PATH)
    recommendations = rec.recommend(USER_ID, top_k=TOP_K)

    print(f"\nTop {TOP_K} recommandations pour user {USER_ID}")
    print(recommendations.round({'score': 4}).to_string())
