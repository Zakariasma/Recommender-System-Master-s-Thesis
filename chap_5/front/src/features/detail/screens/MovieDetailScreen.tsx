import { useEffect, useState, useRef } from "react";
import { useParams } from "react-router";
import { fetchMovieDetails, logMovieView } from "../api/movie_details_api.ts";
import {useMoviesStore} from "../../welcome_grid/stores/useMoviesStore.ts";
import {MovieBackground} from "../components/MovieBackground.tsx";
import {MoviePoster} from "../components/MoviePoster.tsx";
import {MovieInfo} from "../components/MovieInfo.tsx";
import {BackButton} from "../components/BackButton.tsx";

export function MovieDetailScreen() {
    const { id } = useParams<{ id: string }>();
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const detailsCache = useMoviesStore((state) => state.detailsCache);
    const setDetails = useMoviesStore((state) => state.setDetails);

    const movie = id ? detailsCache[id] : undefined;
    const hasFetched = useRef<string | null>(null);

    useEffect(() => {
        if (!id) return;

        if (detailsCache[id]) {
            return;
        }

        if (hasFetched.current === id) return;
        hasFetched.current = id;

        const loadMovie = async () => {
            try {
                setIsLoading(true);
                const data = await fetchMovieDetails(id);
                setDetails(data);
                logMovieView(id);
            } catch (err) {
                console.error(err);
                setError("Impossible de charger le film.");
            } finally {
                setIsLoading(false);
            }
        };

        loadMovie();
    }, [id, detailsCache, setDetails]);

    if (isLoading) return <div className="w-full h-screen bg-rs-black text-rs-white flex items-center justify-center">Chargement...</div>;
    if (error) return <div className="w-full h-screen bg-rs-black text-red-500 flex items-center justify-center">{error}</div>;
    if (!movie) return null;

    return (
        <div className="w-full min-h-screen bg-rs-black text-rs-white relative">
            {movie.background && <MovieBackground src={movie.background} title={movie.title} />}
            <div className="relative z-10 flex flex-col md:flex-row gap-8 p-8 md:p-16 max-w-7xl mx-auto pt-[20vh]">
                {movie.poster && <MoviePoster src={movie.poster} title={movie.title} />}
                <div className="flex flex-col gap-4">
                    <MovieInfo movie={movie} />
                    <BackButton />
                </div>
            </div>
        </div>
    );
}