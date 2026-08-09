import { useEffect, useState } from "react";
import { MovieRowsVertical } from "../components/grid/MovieRowsVertical.tsx";
import { HeroSwiper } from "../components/hero/HeroSwiper.tsx";
import { fetchMovieRows, fetchRecommendations } from "../api/movie_api.ts";
import { useMoviesStore } from "../stores/useMoviesStore.ts";
import type { HeroSlide } from "../types/HeroSlide.ts";

export function WelcomeGridScreen() {
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const rows = useMoviesStore((state) => state.rows);
    const setRows = useMoviesStore((state) => state.setRows);
    const recommendations = useMoviesStore((state) => state.recommendations);
    const setRecommendations = useMoviesStore((state) => state.setRecommendations);

    const defaultHeroSlides: HeroSlide[] = [
        { id: '1', background: '/img/hero1.jpg', title: 'Film à la une 1', description: 'Desc 1', year: 2023, rating: 8.5 },
        { id: '2', background: '/img/hero2.jpg', title: 'Film à la une 2', description: 'Desc 2', year: 2022, rating: 7.5 },
        { id: '3', background: '/img/hero3.jpg', title: 'Film à la une 3', description: 'Desc 3', year: 2021, rating: 9.0 },
    ];

    const heroSlides = recommendations.length > 0 ? recommendations : defaultHeroSlides;

    // 1. Chargement initial des lignes de films (une seule fois)
    useEffect(() => {
        if (rows.length > 0) {
            setIsLoading(false);
            return;
        }

        const loadMovies = async () => {
            try {
                setIsLoading(true);
                const data = await fetchMovieRows();
                setRows(data);
                setError(null);
            } catch (err) {
                console.error(err);
                setError("Impossible de charger les films.");
            } finally {
                setIsLoading(false);
            }
        };

        loadMovies();
    }, [rows.length, setRows]);

    // 2. Récupération des recommandations à chaque fois qu'on arrive sur l'accueil
    useEffect(() => {
        const loadRecs = async () => {
            try {
                const recs = await fetchRecommendations();
                if (recs.length > 0) {
                    setRecommendations(recs);
                }
            } catch (err) {
                console.error("Erreur lors du chargement des recommandations:", err);
            }
        };
        loadRecs();
    }, [setRecommendations]);

    return (
        <div className="w-full h-screen bg-rs-black overflow-hidden overflow-x-hidden">
            {isLoading && <div className="p-4 text-white text-center">Chargement des films...</div>}
            {error && <div className="p-4 text-red-500 text-center">{error}</div>}

            {!isLoading && !error && (
                <MovieRowsVertical
                    rows={rows}
                    hero={<HeroSwiper slides={heroSlides} />}
                />
            )}
        </div>
    );
}