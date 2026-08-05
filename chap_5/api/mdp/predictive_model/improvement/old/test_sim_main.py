from chap_5.api.mdp.predictive_model.helper.encoder import decode


class SimilarityModel:
    def __init__(self, k: int):
        self.k = k
        # connexions : PostgreSQL (transitions), SQLite index, SQLite sortie

    def _candidates(self, self_id, movies):
        """Trouve les voisins (états avec sim > 0) et leur similarité."""
        pass

    def _fetch_state_strings(self, state_ids):
        """Récupère les chaînes d'états à partir de leurs ids."""
        pass

    def _fetch_successors(self, state_strs):
        """Récupère tr_skip(t, s') pour tous les voisins."""
        pass

    def fit(self):
        """Boucle principale : pour chaque état, calcule et sauvegarde tr_sim."""
        states = self.idx_conn.execute(
            "SELECT state_id, state_str FROM states"
        ).fetchall()

        for state_id, state_str in states:
            movies = decode(state_str)
            sims = self._candidates(state_id, movies)


    def _flush(self, rows):
        """Écrit les résultats dans la base de sortie."""
        pass

    def _log(self, done, total, start):
        """Affiche la progression et l'ETA."""
        pass

    def close(self):
        """Ferme les connexions."""
        pass