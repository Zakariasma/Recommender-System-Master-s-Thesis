import { Star } from 'lucide-react'

type MovieBadgesProps = {
    tag?: string
    year: number
    rating: number
}

export function MovieBadges({ tag, year, rating }: MovieBadgesProps) {
    return (
        <div className="flex w-full h-1/10 flex-row items-center">
            {tag && (
                <div className="flex justify-center items-center mr-2 h-3/4 border border-rs-border bg-gray-50/10 backdrop-blur-md rounded-full">
                    <p className="mx-4 text-rs-white">{tag}</p>
                </div>
            )}
            <div className="flex justify-center items-center mr-2 h-3/4 border border-rs-border bg-gray-50/10 backdrop-blur-md rounded-full">
                <p className="mx-4 text-rs-white">{year}</p>
            </div>
            <div className="flex justify-center items-center mr-2 h-3/4 border border-rs-border bg-yellow-300/10 backdrop-blur-md rounded-full px-3">
                <p className="ml-1.5 text-rs-white mr-1">{rating}</p>
                <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
            </div>
        </div>
    )
}