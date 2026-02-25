from chap_1.collaborative_filtering.pearson_recommender import PearsonRecommender
from chap_1.collaborative_filtering.cosine_recommender import CosineRecommender
from chap_1.shared.utils.dataset_retrieve import DatasetRetriever
from chap_1.shared.utils.map_user_history import export_user_movie_ids_csv

USER_ID           = 10
TOP_K             = 500
TOP_DISPLAY       = 15
MIN_PEARSON_SCORE = 3.5
USER_HISTORY_PATH = "data/user_history/user_movie_ids.csv"
MOVIES_PATH       = "../shared/data/movies/movies_dataset_kaggle.csv"

if __name__ == "__main__":
    DatasetRetriever().load_data()
    export_user_movie_ids_csv(USER_HISTORY_PATH, include_rating=True)

    rec_pearson = PearsonRecommender(USER_HISTORY_PATH, MOVIES_PATH)
    rec_cosine  = CosineRecommender(USER_HISTORY_PATH, MOVIES_PATH)

    pearson_recs = rec_pearson.recommend(USER_ID, top_k=TOP_K)
    cosine_recs  = rec_cosine.recommend(USER_ID, top_k=TOP_K)

    print(f"\nTop {TOP_DISPLAY} Pearson pour user {USER_ID}")
    print(pearson_recs.head(TOP_DISPLAY).round({"score": 4}).to_string())

    print(f"\nTop {TOP_DISPLAY} Cosine pour user {USER_ID}")
    print(cosine_recs.head(TOP_DISPLAY).round({"score": 4}).to_string())

    # Filtre Pearson note prédite > 3.5
    pearson_recs = pearson_recs[pearson_recs["score"] > MIN_PEARSON_SCORE]

    # Films recommandés par les deux
    pearson_tmdb = set(pearson_recs["tmdb_id"])
    cosine_tmdb  = set(cosine_recs["tmdb_id"])
    common       = pearson_tmdb & cosine_tmdb

    print(f"\nPearson (score > {MIN_PEARSON_SCORE}) : {len(pearson_tmdb)} films")
    print(f"Cosine  (top {TOP_K})               : {len(cosine_tmdb)} films")
    print(f"Films recommandés par les deux      : {len(common)}")

    print("\nFilms en commun :")
    common_df = pearson_recs[pearson_recs["tmdb_id"].isin(common)][["title", "tmdb_id", "score"]].reset_index(drop=True)
    common_df = common_df.rename(columns={"score": "pearson_score"})
    common_df["cosine_score"] = common_df["tmdb_id"].map(cosine_recs.set_index("tmdb_id")["score"])
    print(common_df.to_string())
