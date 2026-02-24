import os
from chap_1.shared.utils.dataset_retrieve import DatasetRetriever


def pad_imdb_id(imdb_id):
    if not imdb_id or imdb_id == '':
        return ''
    imdb_num = str(imdb_id).lstrip('tt')
    padded = imdb_num.zfill(7)
    return f"tt{padded}"


def export_user_movie_ids_csv(
    output_path: str = "data/user_history/user_movie_ids.csv",
    include_rating: bool = False
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _, ratings, _, links = DatasetRetriever().load_data()

    links['imdbId'] = links['imdbId'].fillna('').apply(pad_imdb_id)
    links['tmdbId'] = links['tmdbId'].fillna('').astype(str)

    links_subset = links[['movieId', 'imdbId', 'tmdbId']]
    merged = ratings.merge(links_subset, on='movieId', how='inner')

    if include_rating:
        merged['rating'] = merged['rating'].clip(0, 5)
        result = merged[['userId', 'imdbId', 'tmdbId', 'rating']]
    else:
        result = merged[['userId', 'imdbId', 'tmdbId']]

    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    export_user_movie_ids_csv()
