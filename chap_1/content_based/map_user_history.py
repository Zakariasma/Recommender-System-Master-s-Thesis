import os
from dataset_retrieve import DatasetRetriever


def pad_imdb_id(imdb_id):
    if not imdb_id or imdb_id == '':
        return ''
    imdb_num = str(imdb_id).lstrip('tt')
    padded = imdb_num.zfill(7)  # Pad à 7 chiffres avec 0
    return f"tt{padded}"


def export_user_movie_ids_csv(output_path: str = "data/user_history/user_movie_ids.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _, ratings, _, links = DatasetRetriever().load_data()

    # imdbId : tt + 7 chiffres minimum -> dataset ML diff de mon dataset donc add pad
    links['imdbId'] = links['imdbId'].fillna('').apply(pad_imdb_id)
    links['tmdbId'] = links['tmdbId'].fillna('').astype(str)

    links_subset = links[['movieId', 'imdbId', 'tmdbId']]
    merged = ratings.merge(links_subset, on='movieId', how='inner')

    result = merged[['userId', 'imdbId', 'tmdbId']]
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    export_user_movie_ids_csv()
