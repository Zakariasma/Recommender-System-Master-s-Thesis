import { useState, useEffect, useRef } from "react";
import { Search } from "lucide-react";
import type { MoviePreview } from "../../../grid/types/MoviePreview.ts";
import { searchMovies } from "../../api/movie_api.ts";
import { SearchDropdown } from "./SearchDropdown.tsx";

export function SearchBar() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<MoviePreview[]>([]);
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const searchRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const timer = setTimeout(async () => {
            if (query.trim().length > 1) {
                setIsLoading(true);
                try {
                    const data = await searchMovies(query);
                    setResults(data);
                    setIsDropdownOpen(true);
                } catch (error) {
                    console.error(error);
                } finally {
                    setIsLoading(false);
                }
            } else {
                setResults([]);
                setIsDropdownOpen(false);
            }
        }, 300);

        return () => clearTimeout(timer);
    }, [query]);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
                setIsDropdownOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const handleSelect = () => {
        setIsDropdownOpen(false);
        setQuery("");
    };

    return (
        <div className="relative w-72" ref={searchRef}>
            <div className="relative flex items-center">
                <Search className="absolute left-3 w-4 h-4 text-gray-400 pointer-events-none" />
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onFocus={() => results.length > 0 && setIsDropdownOpen(true)}
                    placeholder="Rechercher un film..."
                    className="w-full pl-9 pr-3 py-2 bg-rs-border/50 text-rs-white rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-rs-white transition-all"
                />
            </div>

            {isDropdownOpen && (
                <SearchDropdown
                    results={results}
                    isLoading={isLoading}
                    onSelect={handleSelect}
                />
            )}
        </div>
    );
}