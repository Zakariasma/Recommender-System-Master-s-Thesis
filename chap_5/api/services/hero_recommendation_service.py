from sqlalchemy import text, bindparam

from chap_5.api.mdp.config import K
from chap_5.api.mdp.serving.recommender import Recommender
from chap_5.api.schemas.movie import HeroSlide


class HeroRecommendationService:

    def __init__(self, engine, recommender: Recommender):
        self.engine = engine
        self.recommender = recommender

    def get_recommendations(self) -> list[HeroSlide]:
        """Récupère les recommandations MDP basées sur le dernier état de l'historique global."""
        try:
            state = self._build_current_state()
            if state is None:
                return []

            rec_ids = self.recommender.recommend(state)
            if not rec_ids:
                return []

            return self._to_hero_slides(rec_ids)
        except Exception as e:
            print(f"Erreur recommandations MDP: {e}")
            return []

    def _build_current_state(self) -> tuple | None:
        """Construit l'état (derniers K films vus) à partir de l'historique global de l'app."""
        query = text("SELECT movie_id FROM movie_history ORDER BY viewed_at DESC LIMIT :limit")

        with self.engine.connect() as conn:
            rows = conn.execute(query, {"limit": K}).fetchall()

        if not rows:
            return None

        state_list = [r[0] for r in rows]
        while len(state_list) < K:
            state_list.insert(0, state_list[-1])  # bourrage avec le dernier film connu

        return tuple(state_list[:K])

    def _to_hero_slides(self, rec_ids: list) -> list[HeroSlide]:
        """Récupère les détails des films recommandés et les convertit en HeroSlide."""
        query = text("""
            SELECT id, title, background, poster, plot, release_date, score
            FROM movies
            WHERE id IN :ids
        """).bindparams(bindparam("ids", expanding=True))

        with self.engine.connect() as conn:
            rows = conn.execute(query, {"ids": [int(r) for r in rec_ids]}).mappings().all()

        return [self._to_hero_slide(r) for r in rows]

    @staticmethod
    def _to_hero_slide(row) -> HeroSlide:
        year = None
        if row.get("release_date"):
            try:
                year = int(str(row["release_date"])[:4])
            except (ValueError, TypeError):
                pass

        return HeroSlide(
            id=str(row["id"]),
            background=row.get("background") or row.get("poster") or "",
            title=row["title"],
            description=row.get("plot"),
            year=year,
            rating=row.get("score"),
            tag="Recommandé pour vous",
        )