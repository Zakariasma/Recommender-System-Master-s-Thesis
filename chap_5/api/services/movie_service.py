import json
from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine

from chap_5.api.mdp.dataset.utils.fill_up_db import get_engine
from chap_5.api.schemas.movie import MovieRow, MoviePreview, MovieDetails


class MovieService:
    def __init__(self, engine: Engine | None = None):
        self.engine = engine or get_engine()

    def get_genre_rows(self) -> list[MovieRow]:
        cache_rows = self._get_genre_cache()
        if not cache_rows:
            return []

        all_movie_ids = {
            movie_id
            for row in cache_rows
            for movie_id in self._parse_movie_ids(row["movie_ids"])
        }
        movies_by_id = self._get_movies_by_ids(all_movie_ids)

        rows: list[MovieRow] = []
        for row in cache_rows:
            movie_ids = self._parse_movie_ids(row["movie_ids"])
            previews = [
                movies_by_id[movie_id]
                for movie_id in movie_ids
                if movie_id in movies_by_id
            ]
            rows.append(MovieRow(genre_name=row["genre_name"], movies_preview=previews))

        return rows

    def _get_genre_cache(self):
        query = text("""
            SELECT genre_id, genre_name, movie_ids
            FROM genre_movie_cache
            ORDER BY id
        """)
        with self.engine.connect() as conn:
            return conn.execute(query).mappings().all()

    @staticmethod
    def _parse_movie_ids(raw) -> list[int]:
        return json.loads(raw) if isinstance(raw, str) else raw

    def _get_movies_by_ids(self, movie_ids: set[int]) -> dict[int, MoviePreview]:
        if not movie_ids:
            return {}

        query = text("""
            SELECT id, title, poster
            FROM movies
            WHERE id IN :ids
        """).bindparams(bindparam("ids", expanding=True))

        with self.engine.connect() as conn:
            rows = conn.execute(query, {"ids": list(movie_ids)}).mappings().all()

        return {
            row["id"]: MoviePreview(
                id=str(row["id"]),
                background=row["poster"] or "",
                title=row["title"],
            )
            for row in rows
        }

    def get_movie_details(self, movie_id: int) -> MovieDetails | None:
        query = text("""
                     SELECT id,
                            logo_title,
                            country,
                            duration,
                            score,
                            plot,
                            release_date,
                            background,
                            title,
                            poster
                     FROM movies
                     WHERE id = :movie_id
                     """)
        with self.engine.connect() as conn:
            row = conn.execute(query, {"movie_id": movie_id}).mappings().first()

        if not row:
            return None

        return MovieDetails(
            id=str(row["id"]),
            logo_title=row.get("logo_title"),
            country=row.get("country"),
            duration=row.get("duration"),
            score=row.get("score"),
            plot=row.get("plot"),
            release_date=str(row.get("release_date")) if row.get("release_date") else None,
            background=row.get("background"),
            title=row["title"],
            poster=row.get("poster")
        )

    def add_to_history(self, movie_id: int):
        query = text("INSERT INTO movie_history (movie_id) VALUES (:movie_id)")
        with self.engine.begin() as conn:
            conn.execute(query, {"movie_id": movie_id})

    def search_movies(self, query: str, limit: int = 5) -> list[MoviePreview]:
        sql_query = text("""
                         SELECT id, title, poster
                         FROM movies
                         WHERE LOWER(title) LIKE LOWER(:query)
                         ORDER BY (wikiquote_sitelinks + wikipedia_sitelinks) DESC LIMIT :limit
                         """)

        search_term = f"%{query}%"

        with self.engine.connect() as conn:
            rows = conn.execute(sql_query, {"query": search_term, "limit": limit}).mappings().all()

        return [
            MoviePreview(
                id=str(row["id"]),
                background=row["poster"] or "",
                title=row["title"]
            )
            for row in rows
        ]

    def get_history(self, limit: int = 20) -> list[MoviePreview]:
        query = text("""
                     SELECT m.id, m.title, m.poster, mh.viewed_at
                     FROM movie_history mh
                              JOIN movies m ON m.id = mh.movie_id
                     ORDER BY mh.viewed_at DESC LIMIT :limit
                     """)

        with self.engine.connect() as conn:
            rows = conn.execute(query, {"limit": limit}).mappings().all()

        return [
            MoviePreview(
                id=str(row["id"]),
                background=row["poster"] or "",
                title=row["title"]
            )
            for row in rows
        ]

    @staticmethod
    def _to_movie_preview(row) -> MoviePreview:
        return MoviePreview(
            id=str(row["id"]),
            background=row["poster"] or "",
            title=row["title"],
        )