import json

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    bindparam,
    func,
    text,
)

from chap_5.api.mdp.dataset.collumn_setup.seed_database import get_engine


TOP_GENRES_COUNT = 7
MOVIES_PER_GENRE = 20

metadata = MetaData()

genre_movie_cache = Table(
    "genre_movie_cache",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("genre_id", Integer, nullable=False, unique=True),
    Column("genre_name", String, nullable=False),
    Column("movie_ids", JSON, nullable=False),
    Column(
        "updated_at",
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    ),
)


def create_cache_table(engine):
    metadata.create_all(engine, tables=[genre_movie_cache])


def clear_cache(engine):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM genre_movie_cache"))


def get_top_genres(engine, limit: int = TOP_GENRES_COUNT):
    query = text(
        """
        SELECT
            g.id AS genre_id,
            g.name AS genre_name,
            COUNT(mg.movie_id) AS movie_count
        FROM genres g
        JOIN movie_genre mg ON mg.genre_id = g.id
        GROUP BY g.id, g.name
        ORDER BY movie_count DESC
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        result = conn.execute(query, {"limit": limit})
        return result.mappings().all()


def get_movie_ids_for_genre(
    engine,
    genre_id: int,
    limit: int = MOVIES_PER_GENRE,
    exclude_ids: set | None = None,
):
    exclude_ids = exclude_ids or set()

    base_query = """
        SELECT mg.movie_id
        FROM movie_genre mg
        JOIN movies m ON m.id = mg.movie_id
        WHERE mg.genre_id = :genre_id
    """

    order_by_query = """
        ORDER BY
            (m.wikiquote_sitelinks + m.wikipedia_sitelinks) DESC,
            m.release_date DESC
        LIMIT :limit
    """

    if exclude_ids:
        query = (
            text(
                base_query
                + " AND mg.movie_id NOT IN :exclude_ids "
                + order_by_query
            )
            .bindparams(bindparam("exclude_ids", expanding=True))
        )

        params = {
            "genre_id": genre_id,
            "limit": limit,
            "exclude_ids": list(exclude_ids),
        }
    else:
        query = text(base_query + order_by_query)
        params = {
            "genre_id": genre_id,
            "limit": limit,
        }

    with engine.connect() as conn:
        result = conn.execute(query, params)
        return [row[0] for row in result]


def init_genre_cache():
    engine = get_engine()

    create_cache_table(engine)
    clear_cache(engine)

    top_genres = get_top_genres(engine)
    used_movie_ids = set()

    with engine.begin() as conn:
        for genre in top_genres:
            movie_ids = get_movie_ids_for_genre(
                engine,
                genre["genre_id"],
                exclude_ids=used_movie_ids,
            )

            used_movie_ids.update(movie_ids)

            conn.execute(
                genre_movie_cache.insert().values(
                    genre_id=genre["genre_id"],
                    genre_name=genre["genre_name"],
                    movie_ids=json.dumps(movie_ids),
                )
            )


if __name__ == "__main__":
    init_genre_cache()