from chap_1.content_based.services.init_dataset import DatasetRetriever
from chap_1.content_based.services.recommender import ContentBasedRecommender
from chap_1.content_based.services.tfidf_matrix import TfidfMatrixBuilder
from chap_1.content_based.services.user_profile import UserProfileBuilder

if __name__ == "__main__":
    retriever = DatasetRetriever()
    ratings, movies = retriever.load_data()

    builder = TfidfMatrixBuilder()
    tfidf_matrix = builder.build(movies)

    user_id = 10
    user_builder = UserProfileBuilder(user_id, ratings, movies, tfidf_matrix)

    user_builder.show_last_watched()

    user_vector = user_builder.build_profile_vector()

    if user_vector is not None:
        watched_movie_ids = set(ratings[ratings['userId'] == user_id]['movieId'].values)

        recommender = ContentBasedRecommender(tfidf_matrix, movies, user_vector, top_k=50)
        recommender.show_recommendations(watched_movie_ids)
    else:
        print("Aucun film aimé ou aucun film visionné, impossible de construire un profil.")