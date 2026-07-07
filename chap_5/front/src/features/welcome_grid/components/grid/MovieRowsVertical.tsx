import { useRef } from 'react'
import type { ReactNode } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { MovieRow } from '../../types/MovieRow.ts'
import { MovieRowHorizontal } from './MovieRowHorizontal.tsx'
import { useViewportHeight } from "../../hooks/useViewportHeight.ts"
import {Navbar} from "../../../shared/components/navbar/Navbar.tsx";

type MovieRowsVerticalProps = {
    rows: MovieRow[]
    rowsPerScreen?: number
    hero?: ReactNode
}

export function MovieRowsVertical({ rows, rowsPerScreen = 3, hero }: MovieRowsVerticalProps) {
    const parentRef = useRef<HTMLDivElement>(null)
    const viewportHeight = useViewportHeight()
    const rowHeight = viewportHeight / rowsPerScreen

    const virtualizer = useVirtualizer({
        count: rows.length,
        getScrollElement: () => parentRef.current,
        estimateSize: () => rowHeight,
        overscan: 2,
    })

    return (
        <div ref={parentRef} className="w-full h-screen overflow-y-auto overflow-x-hidden bg-rs-black">
            <Navbar />

            {hero && (
                <div className="w-full">
                    {hero}
                </div>
            )}

            <div
                style={{
                    height: `${virtualizer.getTotalSize()}px`,
                    width: '100%',
                    position: 'relative',
                }}
            >
                {virtualizer.getVirtualItems().map((virtualItem) => {
                    const row = rows[virtualItem.index]
                    return (
                        <div
                            key={virtualItem.key}
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: `${rowHeight}px`,
                                transform: `translateY(${virtualItem.start}px)`,
                            }}
                        >
                            <MovieRowHorizontal
                                title={row.genre_name}
                                movies={row.movies_preview}
                                rowHeight={rowHeight}
                            />
                        </div>
                    )
                })}
            </div>
        </div>
    )
}