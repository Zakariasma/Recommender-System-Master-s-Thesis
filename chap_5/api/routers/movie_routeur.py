from fastapi import APIRouter, Depends, HTTPException

from chap_5.api.schemas.movie import MovieRow, MovieDetails, MoviePreview
from chap_5.api.services.movie_service import MovieService

router = APIRouter(prefix="/movies", tags=["movies"])


def get_movie_service() -> MovieService:
    return MovieService()


@router.get("/rows", response_model=list[MovieRow])
def get_movie_rows(service: MovieService = Depends(get_movie_service)):
    return service.get_genre_rows()


# 1. LA ROUTE SEARCH EN PREMIER (avant l'ID)
@router.get("/search", response_model=list[MoviePreview])
def search_movies(query: str, service: MovieService = Depends(get_movie_service)):
    if not query:
        return []
    return service.search_movies(query)


# 2. ENSUITE LA ROUTE AVEC L'ID
@router.get("/{movie_id}", response_model=MovieDetails)
def get_movie_details(movie_id: int, service: MovieService = Depends(get_movie_service)):
    details = service.get_movie_details(movie_id)
    if not details:
        raise HTTPException(status_code=404, detail="Film non trouvé")
    return details


@router.post("/{movie_id}/view")
def view_movie(movie_id: int, service: MovieService = Depends(get_movie_service)):
    try:
        service.add_to_history(movie_id)
        return {"message": "Film ajouté à l'historique"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))