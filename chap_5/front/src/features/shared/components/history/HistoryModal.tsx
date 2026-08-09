import { useEffect } from "react";
import { X } from "lucide-react";
import { MovieRowHorizontal } from "../../../grid/components/grid/MovieRowHorizontal.tsx";
import type { MoviePreview } from "../../../grid/types/MoviePreview.ts";

type HistoryModalProps = {
    isOpen: boolean;
    onClose: () => void;
    movies: MoviePreview[];
    isLoading: boolean;
};

const ROW_HEIGHT = 300;

export function HistoryModal({ isOpen, onClose, movies, isLoading }: HistoryModalProps) {
    useEffect(() => {
        if (!isOpen) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        document.addEventListener("keydown", handleKeyDown);
        document.body.style.overflow = "hidden";

        return () => {
            document.removeEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "";
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-[100] flex items-center justify-center w-full h-screen backdrop-blur-xl"
            onClick={onClose}
        >
            <div
                className="w-[90%] max-w-5xl h-1/3 bg-rs-black border border-rs-border rounded-xl shadow-2xl overflow-hidden flex flex-col"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="flex items-center justify-between px-6 py-3 border-b border-rs-border flex-shrink-0">
                    <h2 className="text-rs-white text-lg font-semibold">Historique</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-rs-white transition-colors cursor-pointer"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="flex-1 overflow-hidden">
                    {isLoading ? (
                        <div className="flex items-center justify-center h-full text-sm text-gray-400">
                            Chargement...
                        </div>
                    ) : movies.length > 0 ? (
                        <MovieRowHorizontal title="" movies={movies} rowHeight={ROW_HEIGHT} dimAfterIndex={3} />
                    ) : (
                        <div className="flex items-center justify-center h-full text-sm text-gray-400">
                            Aucun historique pour l'instant
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}