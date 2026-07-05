import json
from sqlalchemy import MetaData, Table, Column, Integer, String, JSON, DateTime, func, text, bindparam

from chap_5.api.mdp.dataset.seed_database import get_engine

TOP_GENRES_COUNT = 7
MOVIES_PER_GENRE = 20

metadata = MetaData()

genre_movie_cache = Table(
    'genre_movie_cache',
    metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('genre_id', Integer, nullable=False, unique=True),
    Column('genre_name', String, nullable=False),
    Column('movie_ids', JSON, nullable=False),
    Column('updated_at', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
)


def create_cache_table(engine):
    metadata.create_all(engine, tables=[genre_movie_cache])


def clear_cache(engine):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM genre_movie_cache"))


def get_top_genres(engine, limit: int = TOP_GENRES_COUNT):
    """Retourne les genres ayant le plus de films associés (via movie_genre)."""
    query = text("""
                 SELECT g.id AS genre_id, g.name AS genre_name, COUNT(mg.movie_id) AS movie_count
                 FROM genres g
                          JOIN movie_genre mg ON mg.genre_id = g.id
                 GROUP BY g.id, g.name
                 ORDER BY movie_count DESC LIMIT :limit
                 """)
    with engine.connect() as conn:
        result = conn.execute(query, {"limit": limit})
        return result.mappings().all()


def get_movie_ids_for_genre(engine, genre_id: int, limit: int = MOVIES_PER_GENRE, exclude_ids: set = None):
    """
    Retourne jusqu'à `limit` ids de films pour un genre donné.
    Trie par popularité (somme des sitelinks) puis par date de sortie (plus récents en premier).
    Exclut les films dont les ids sont présents dans `exclude_ids`.
    """
    if exclude_ids is None:
        exclude_ids = set()

    # Base de la requête avec JOIN sur la table movies
    base_query = """
                 SELECT mg.movie_id
                 FROM movie_genre mg
                          JOIN movies m ON m.id = mg.movie_id
                 WHERE mg.genre_id = :genre_id \
                 """

    # Tri : d'abord par la somme des sitelinks (décroissant), puis par la date (décroissant)
    order_by_query = """
        ORDER BY (m.wikiquote_sitelinks + m.wikipedia_sitelinks) DESC, m.release_date DESC
        LIMIT :limit
    """

    if exclude_ids:
        # Si on a des films à exclure, on ajoute la condition NOT IN
        query = text(base_query + " AND mg.movie_id NOT IN :exclude_ids " + order_by_query) \
            .bindparams(bindparam("exclude_ids", expanding=True))

        params = {"genre_id": genre_id, "limit": limit, "exclude_ids": list(exclude_ids)}
    else:
        # Requête normale s'il n'y a aucun film à exclure (première catégorie)
        query = text(base_query + order_by_query)
        params = {"genre_id": genre_id, "limit": limit}

    with engine.connect() as conn:
        result = conn.execute(query, params)
        return [row[0] for row in result]


def init_genre_cache():
    engine = get_engine()

    print("Création de la table de cache si nécessaire...")
    create_cache_table(engine)

    print("Nettoyage de l'ancien cache...")
    clear_cache(engine)

    print(f"Récupération des {TOP_GENRES_COUNT} genres les plus populaires...")
    top_genres = get_top_genres(engine)

    # Set pour garder en mémoire les films déjà placés dans d'autres genres
    used_movie_ids = set()

    with engine.begin() as conn:
        for genre in top_genres:
            # On récupère les films en excluant ceux déjà utilisés
            movie_ids = get_movie_ids_for_genre(
                engine,
                genre["genre_id"],
                exclude_ids=used_movie_ids
            )

            # On ajoute les nouveaux films à l'ensemble pour les exclure des prochains genres
            used_movie_ids.update(movie_ids)

            print(f"  - {genre['genre_name']} : {len(movie_ids)} films")

            conn.execute(
                genre_movie_cache.insert().values(
                    genre_id=genre["genre_id"],
                    genre_name=genre["genre_name"],
                    movie_ids=json.dumps(movie_ids),
                )
            )

    print("Cache des genres initialisé.")


if __name__ == "__main__":
    init_genre_cache()