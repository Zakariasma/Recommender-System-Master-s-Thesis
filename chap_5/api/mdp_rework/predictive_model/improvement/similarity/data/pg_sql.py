from collections import defaultdict
from sqlalchemy import create_engine, text
from chap_5.api.mdp.config import DATABASE_URL


def get_pg_engine():
    return create_engine(DATABASE_URL)


def fetch_movie_genres(engine) -> dict:
    movie_genres = defaultdict(set)
    with engine.connect() as conn:
        for movie_id, genre_id in conn.execute(text("SELECT movie_id, genre_id FROM movie_genre")):
            movie_genres[int(movie_id)].add(int(genre_id))
    return dict(movie_genres)