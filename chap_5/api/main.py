# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chap_5.api.routers.movie_routeur import router as movie_routeur

app = FastAPI()

# Configuration du CORS
origins = [
    "http://localhost:5173",  # Remplacez par le port de votre app React (Vite est souvent 5173, CRA est 3000)
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(movie_routeur)