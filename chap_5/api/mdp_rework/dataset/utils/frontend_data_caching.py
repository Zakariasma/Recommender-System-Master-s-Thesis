import json
from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, bindparam, func, text
from chap_5.api.mdp.dataset.collumn_setup.seed_database import get_engine

TOP_GENRES_COUNT = 7
MOVIES_PER_GENRE = 20

metadata = MetaData()

genre_movie_cache = Table(
    "genre_movie_cache", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("genre_id", Integer, nullable=False, unique=True),
    Column("genre_name", String, nullable=False),
    Column("movie_ids", JSON, nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)


def init_frontend_cache():
    engine = get_engine()
    metadata.create_all(engine, tables=[genre_movie_cache])

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM genre_movie_cache"))

        top_genres = conn.execute(text("""
                                       SELECT g.id AS genre_id, g.name AS genre_name, COUNT(mg.movie_id) AS movie_count
                                       FROM genres g
                                                JOIN movie_genre mg ON mg.genre_id = g.id
                                       GROUP BY g.id, g.name
                                       ORDER BY movie_count DESC LIMIT :limit
                                       """), {"limit": TOP_GENRES_COUNT}).mappings().all()

        used_ids = set()

        for g in top_genres:
            q_str = "SELECT mg.movie_id FROM movie_genre mg JOIN movies m ON m.id = mg.movie_id WHERE mg.genre_id = :gid"
            params = {"gid": g["genre_id"], "limit": MOVIES_PER_GENRE}

            if used_ids:
                q_str += " AND mg.movie_id NOT IN :ex"
                params["ex"] = list(used_ids)

            q_str += " ORDER BY (m.wikiquote_sitelinks + m.wikipedia_sitelinks) DESC, m.release_date DESC LIMIT :limit"

            query = text(q_str)
            if used_ids:
                query = query.bindparams(bindparam("ex", expanding=True))

            movie_ids = [r[0] for r in conn.execute(query, params)]
            used_ids.update(movie_ids)

            conn.execute(genre_movie_cache.insert().values(
                genre_id=g["genre_id"],
                genre_name=g["genre_name"],
                movie_ids=json.dumps(movie_ids)
            ))


if __name__ == "__main__":
    init_frontend_cache()