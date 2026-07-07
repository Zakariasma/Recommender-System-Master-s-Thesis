import { useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { MoviePreview } from '../../types/MoviePreview.ts'
import { getCardDimensions } from '../../utils/layout.ts'
import { MovieCard } from './MovieCard.tsx'
import { ScrollRightButton } from './ScrollRightButton.tsx'

type MovieRowHorizontalProps = {
    title: string
    movies: MoviePreview[]
    rowHeight: number
    dimAfterIndex?: number
}

export function MovieRowHorizontal({ title, movies, rowHeight, dimAfterIndex }: MovieRowHorizontalProps) {
    const parentRef = useRef<HTMLDivElement>(null)
    const { cardWidth, cardHeight } = getCardDimensions(rowHeight)

    const virtualizer = useVirtualizer({
        count: movies.length,
        getScrollElement: () => parentRef.current,
        horizontal: true,
        estimateSize: () => cardWidth,
        overscan: 5,
        gap: 16,
    })

    const scrollRight = () => {
        parentRef.current?.scrollBy({ left: cardWidth * 3, behavior: 'smooth' })
    }

    const formattedTitle = title.charAt(0).toUpperCase() + title.slice(1)

    return (
        <div className="my-2 w-[97%] mx-auto">
            {title && <h2 className="text-rs-white text-xl font-semibold mb-3">{formattedTitle}</h2>}

            <div className="relative w-full">
                <div
                    ref={parentRef}
                    className="overflow-x-auto [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
                >
                    <div
                        style={{
                            width: `${virtualizer.getTotalSize()}px`,
                            height: `${cardHeight}px`,
                            position: 'relative',
                        }}
                    >
                        {virtualizer.getVirtualItems().map((virtualItem) => {
                            const movie = movies[virtualItem.index]
                            const isDimmed = dimAfterIndex !== undefined && virtualItem.index >= dimAfterIndex

                            return (
                                <div
                                    key={virtualItem.key}
                                    className={`absolute top-0 h-full transition-opacity ${isDimmed ? 'opacity-40' : 'opacity-100'}`}
                                    style={{
                                        width: `${virtualItem.size}px`,
                                        transform: `translateX(${virtualItem.start}px)`,
                                    }}
                                >
                                    <MovieCard movie={movie} />
                                </div>
                            )
                        })}
                    </div>
                </div>

                <ScrollRightButton onClick={scrollRight} />
            </div>
        </div>
    )
}