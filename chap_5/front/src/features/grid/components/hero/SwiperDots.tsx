type SwiperDotsProps = {
    count: number
    selectedIndex: number
    onSelect: (index: number) => void
}

export function SwiperDots({ count, selectedIndex, onSelect }: SwiperDotsProps) {
    return (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex flex-row gap-2 z-20">
            {Array.from({ length: count }).map((_, index) => (
                <button
                    key={index}
                    onClick={() => onSelect(index)}
                    className={`h-3 rounded-full cursor-pointer transition-all ${
                        index === selectedIndex ? 'w-10 bg-rs-white' : 'w-3 bg-rs-white/40'
                    }`}
                    aria-label={`Aller au slide ${index + 1}`}
                />
            ))}
        </div>
    )
}