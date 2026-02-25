from chap_1.collaborative_filtering.pearson_recommender import PearsonRecommender
from chap_1.shared.utils.dataset_retrieve import DatasetRetriever
from chap_1.shared.utils.map_user_history import export_user_movie_ids_csv

USER_ID           = 10
TOP_K             = 20
USER_HISTORY_PATH = "data/user_history/user_movie_ids.csv"
MOVIES_PATH       = "../shared/data/movies/movies_dataset_kaggle.csv"

if __name__ == "__main__":
    DatasetRetriever().load_data()
    export_user_movie_ids_csv(USER_HISTORY_PATH, include_rating=True)

    print("\nPEARSON")
    rec_pearson = PearsonRecommender(USER_HISTORY_PATH, MOVIES_PATH)
    rec_pearson.print_user_history(USER_ID)
    print(f"\nTop {TOP_K} pour user {USER_ID} (Pearson)")
    print(rec_pearson.recommend(USER_ID, top_k=TOP_K).round({"score": 4}).to_string())

