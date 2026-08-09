import { Link } from "react-router";
import type { MoviePreview } from "../../../grid/types/MoviePreview.ts";

type SearchDropdownProps = {
    results: MoviePreview[];
    isLoading: boolean;
    onSelect: () => void;
};

export function SearchDropdown({ results, isLoading, onSelect }: SearchDropdownProps) {
    return (
        <div className="absolute top-full right-0 mt-2 w-full bg-rs-black/95 backdrop-blur-md rounded-md border border-rs-border overflow-hidden shadow-2xl">
            {isLoading ? (
                <div className="p-4 text-sm text-gray-400 text-center">Recherche...</div>
            ) : results.length > 0 ? (
                results.map((movie) => (
                    <Link
                        key={movie.id}
                        to={`/movie/${movie.id}`}
                        onClick={onSelect}
                        className="flex items-center gap-3 p-2 hover:bg-rs-border transition-colors cursor-pointer"
                    >
                        <img
                            src={movie.background}
                            alt={movie.title}
                            className="w-10 h-14 object-cover rounded-sm flex-shrink-0"
                        />
                        <span className="text-rs-white text-sm truncate">{movie.title}</span>
                    </Link>
                ))
            ) : (
                <div className="p-4 text-sm text-gray-400 text-center">Aucun film trouvé</div>
            )}
        </div>
    );
}