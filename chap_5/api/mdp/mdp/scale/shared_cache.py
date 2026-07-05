# chap_5/api/mdp/mdp/scale/shared_cache.py
import pandas as pd
from sqlalchemy import create_engine, text
from chap_5.api.mdp.config import DATABASE_URL

# On initialise le dictionnaire vide dès le départ
_global_trans_cache = {}


def load_transitions_to_ram():
    global _global_trans_cache
    print("Préchargement des transitions (dictionnaire optimisé)...")
    engine = create_engine(DATABASE_URL)

    # On vide le dictionnaire existant et on ajoute les clés 1, 2, 3
    # Surtout PAS de _global_trans_cache = {...} ici !
    _global_trans_cache.clear()
    _global_trans_cache[1] = {}
    _global_trans_cache[2] = {}
    _global_trans_cache[3] = {}

    with engine.connect() as conn:
        for k in [1, 2, 3]:
            print(f"  Chargement k={k}...")
            query = text("SELECT s, s_, prob FROM transitions WHERE k = :k")

            for chunk_df in pd.read_sql(query, conn, params={"k": k}, chunksize=200000):
                items = chunk_df['s_'].str.split(',').str[-1].astype(int)

                for s, item, prob in zip(chunk_df['s'], items, chunk_df['prob']):
                    if s not in _global_trans_cache[k]:
                        _global_trans_cache[k][s] = {}
                    _global_trans_cache[k][s][item] = prob

    print("  Transitions chargées en RAM !")