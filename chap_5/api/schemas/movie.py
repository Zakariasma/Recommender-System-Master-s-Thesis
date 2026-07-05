from pydantic import BaseModel
from typing import Optional

class MoviePreview(BaseModel):
    id: str
    background: str
    title: str

class MovieRow(BaseModel):
    genre_name: str
    movies_preview: list[MoviePreview]

class MovieDetails(BaseModel):
    id: str
    logo_title: Optional[str] = None
    country: Optional[str] = None
    duration: Optional[int] = None
    score: Optional[float] = None
    plot: Optional[str] = None
    release_date: Optional[str] = None
    background: Optional[str] = None
    title: str
    poster: Optional[str] = None