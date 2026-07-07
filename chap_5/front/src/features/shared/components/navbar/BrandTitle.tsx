import { Link } from "react-router";

export function BrandTitle() {
    return (
        <Link
            to="/"
            className="text-rs-white text-2xl font-bold tracking-wider hover:text-gray-300 transition-colors"
        >
            RecSys
        </Link>
    );
}