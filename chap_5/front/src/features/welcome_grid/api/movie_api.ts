import type { MovieRow } from "../types/MovieRow.ts";
import type { HeroSlide } from "../types/HeroSlide.ts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function fetchMovieRows(): Promise<MovieRow[]> {
    const response = await fetch(`${API_BASE_URL}/movies/rows`);
    if (!response.ok) throw new Error("Erreur lors de la récupération des lignes de films");
    return response.json();
}

export async function fetchRecommendations(): Promise<HeroSlide[]> {
    const response = await fetch(`${API_BASE_URL}/movies/recommendations`);
    if (!response.ok) throw new Error("Erreur lors de la récupération des recommandations");
    return response.json();
}