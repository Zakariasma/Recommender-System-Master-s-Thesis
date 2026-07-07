import {GOLDEN_RATIO} from "../../shared/utils/layout.ts";

type MoviePosterProps = {
    src: string
    title: string
}

export function MoviePoster({ src, title }: MoviePosterProps) {
    const width = "18rem"
    const height = `calc(18rem * ${GOLDEN_RATIO})`

    return (
        <div
            className="flex-shrink-0 rounded-lg overflow-hidden shadow-2xl border-2 border-rs-border"
            style={{ width, height }}
        >
            <img src={src} alt={title} className="w-full h-full object-cover" />
        </div>
    )
}