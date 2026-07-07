import { BrandTitle } from "./BrandTitle.tsx";
import { HistoryLink } from "./HistoryLink.tsx";
import { SearchBar } from "./SearchBar.tsx";

export function Navbar() {
    return (
        <nav className="sticky top-0 z-50 w-full bg-rs-black/50 backdrop-blur-md border-b border-rs-border">
            <div className="flex items-center justify-between w-[97%] mx-auto h-16">

                <div className="flex items-center gap-8">
                    <BrandTitle />
                    <HistoryLink />
                </div>

                <SearchBar />
            </div>
        </nav>
    );
}