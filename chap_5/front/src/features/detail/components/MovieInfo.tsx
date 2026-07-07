import type {MovieDetails} from "../types/MovieDetails.ts";

type MovieInfoProps = {
    movie: MovieDetails
}

export function MovieInfo({ movie }: MovieInfoProps) {
    return (
        <div className="flex flex-col gap-4">
            {movie.logo_title ? (
                <img src={movie.logo_title} alt={movie.title} className="w-2/3 md:w-1/2 mb-4" />
            ) : (
                <h1 className="text-4xl md:text-6xl font-bold mb-4">{movie.title}</h1>
            )}

            <div className="flex items-center gap-4 text-sm text-gray-300">
                {movie.release_date && <span>{movie.release_date.split('-')[0]}</span>}
                {movie.duration && <span>{Math.floor(movie.duration / 60)}h {movie.duration % 60}min</span>}
                {movie.country && <span>{movie.country}</span>}
                {movie.score !== null && (
                    <span className="text-green-400 font-semibold">⭐ {movie.score}/100</span>
                )}
            </div>

            {movie.plot && (
                <p className="text-gray-200 text-lg max-w-3xl leading-relaxed mt-4">
                    {movie.plot}
                </p>
            )}
        </div>
    )
}