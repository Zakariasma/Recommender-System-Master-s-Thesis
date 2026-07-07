from fastapi import APIRouter, Depends, HTTPException
from chap_5.api.schemas.movie import MovieRow, MovieDetails, MoviePreview, HeroSlide
from chap_5.api.services.movie_service import MovieService
from chap_5.api.services.hero_recommendation_service import HeroRecommendationService
from chap_5.api.mdp.predictive_model.predictive_model import PredictiveModel
from chap_5.api.mdp.serving.recommender import Recommender

router = APIRouter(prefix="/movies", tags=["movies"])


def get_movie_service() -> MovieService:
    return MovieService()


def get_hero_recommendation_service(
    service: MovieService = Depends(get_movie_service),
) -> HeroRecommendationService:
    predictive_model = PredictiveModel()
    recommender = Recommender(predictive_model)
    return HeroRecommendationService(engine=service.engine, recommender=recommender)


@router.get("/rows", response_model=list[MovieRow])
def get_movie_rows(service: MovieService = Depends(get_movie_service)):
    return service.get_genre_rows()


@router.get("/search", response_model=list[MoviePreview])
def search_movies(query: str, service: MovieService = Depends(get_movie_service)):
    if not query:
        return []
    return service.search_movies(query)

@router.get("/history", response_model=list[MoviePreview])
def get_history(service: MovieService = Depends(get_movie_service)):
    return service.get_history()

@router.get("/recommendations", response_model=list[HeroSlide])
def get_recommendations(
    service: HeroRecommendationService = Depends(get_hero_recommendation_service),
):
    return service.get_recommendations()


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