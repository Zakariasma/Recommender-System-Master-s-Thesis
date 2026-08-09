import { useState } from "react";
import type {MoviePreview} from "../../../grid/types/MoviePreview.ts";
import {getHistory} from "../../api/movie_api.ts";
import {HistoryModal} from "../history/HistoryModal.tsx";


export function HistoryLink() {
    const [isOpen, setIsOpen] = useState(false);
    const [movies, setMovies] = useState<MoviePreview[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    const handleOpen = async () => {
        setIsOpen(true);
        setIsLoading(true);
        try {
            const data = await getHistory();
            setMovies(data);
        } catch (error) {
            console.error(error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            <button
                onClick={handleOpen}
                className="flex items-center gap-2 text-gray-400 hover:text-rs-white transition-colors cursor-pointer"
            >
                <span className="hidden md:block text-sm font-medium">Historique</span>
            </button>

            <HistoryModal
                isOpen={isOpen}
                onClose={() => setIsOpen(false)}
                movies={movies}
                isLoading={isLoading}
            />
        </>
    );
}