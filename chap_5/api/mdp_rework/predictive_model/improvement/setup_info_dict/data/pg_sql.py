from sqlalchemy import create_engine, text
from chap_5.api.mdp.config import DATABASE_URL


def get_pg_engine():
    return create_engine(DATABASE_URL)


def fetch_movie_scores(engine) -> dict:
    movie_scores = {}
    with engine.connect() as conn:
        for movie_id, score in conn.execute(text("SELECT id, score FROM movies")):
            movie_scores[int(movie_id)] = float(score)
    return movie_scores