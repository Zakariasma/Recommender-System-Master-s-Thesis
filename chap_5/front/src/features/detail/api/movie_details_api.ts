import type { MovieDetails } from "../types/MovieDetails.ts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function fetchMovieDetails(movieId: string): Promise<MovieDetails> {
    const response = await fetch(`${API_BASE_URL}/movies/${movieId}`);
    if (!response.ok) {
        throw new Error("Erreur lors de la récupération des détails du film");
    }
    return response.json();
}

export async function logMovieView(movieId: string): Promise<void> {
    await fetch(`${API_BASE_URL}/movies/${movieId}/view`, {
        method: "POST",
    });
}