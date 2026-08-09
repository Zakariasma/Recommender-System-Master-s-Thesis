import type {MoviePreview} from "../../grid/types/MoviePreview.ts";


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function searchMovies(query: string): Promise<MoviePreview[]> {
    const response = await fetch(`${API_BASE_URL}/movies/search?query=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error("Erreur lors de la recherche");
    return response.json();
}

export async function getHistory(): Promise<MoviePreview[]> {
    const res = await fetch(`${API_BASE_URL}/movies/history`);
    if (!res.ok) throw new Error("Erreur lors de la récupération de l'historique");
    return res.json();
}