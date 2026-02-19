import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_PORT = os.getenv('DB_PORT', '5432')
required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Variables manquantes dans .env : {missing_vars}")

# SQLAlchemy
connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)


query = """
SELECT 
    m.id,
    m.title,
    m.imdb_id,
    m.tmdb_id,
    m.plot,
    COALESCE(
        json_agg(DISTINCT g.name ORDER BY g.name) FILTER (WHERE g.name IS NOT NULL),
        '[]'::json
    ) as genres,
    COALESCE(
        json_agg(DISTINCT t.name ORDER BY t.name) FILTER (WHERE t.name IS NOT NULL),
        '[]'::json
    ) as themes
FROM movie m
LEFT JOIN movie_genre mg ON m.id = mg.movie_id
LEFT JOIN genre g ON mg.genre_id = g.id
LEFT JOIN movie_theme mt ON m.id = mt.movie_id
LEFT JOIN theme t ON mt.theme_id = t.id
GROUP BY m.id, m.title, m.imdb_id, m.tmdb_id, m.plot
ORDER BY m.title
"""


df = pd.read_sql_query(query, engine)

df['genres'] = df['genres'].apply(lambda x: x if x else [])
df['themes'] = df['themes'].apply(lambda x: x if x else [])

output_file = 'movies_dataset_kaggle.csv'
df.to_csv(output_file, index=False)
