import { useState } from 'react'
import { Link } from 'react-router'
import type { MoviePreview } from '../../types/MoviePreview.ts'

type MovieCardProps = {
    movie: MoviePreview
}

export function MovieCard({ movie }: MovieCardProps) {
    const [isLoaded, setIsLoaded] = useState(false)

    return (
        <Link
            to={`/movie/${movie.id}`}
            className="w-full h-full block rounded-md overflow-hidden bg-rs-border cursor-pointer transition-all duration-200 ease-out scale-95 hover:scale-100 border-2 border-transparent hover:border-rs-white"
        >
            {!isLoaded && (
                <div className="absolute inset-0 w-full h-full bg-rs-border animate-pulse" />
            )}

            <img
                src={movie.background}
                alt={movie.title}
                className={`w-full h-full object-cover transition-opacity duration-150 ${isLoaded ? 'opacity-100' : 'opacity-0'}`}
                onLoad={() => setIsLoaded(true)}
            />

            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-rs-black/80 to-transparent p-2">
                <p className="text-rs-white text-xs font-medium truncate">{movie.title}</p>
            </div>
        </Link>
    )
}