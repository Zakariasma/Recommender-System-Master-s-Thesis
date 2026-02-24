from chap_1.shared.utils.dataset_retrieve import DatasetRetriever
from chap_1.shared.utils.map_user_history import export_user_movie_ids_csv


USER_HISTORY_PATH = "data/user_history/user_movie_ids.csv"

if __name__ == "__main__":
    DatasetRetriever().load_data()
    export_user_movie_ids_csv(USER_HISTORY_PATH)