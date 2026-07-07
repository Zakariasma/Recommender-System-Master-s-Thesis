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
                    className={`h-1.5 rounded-full transition-all ${
                        index === selectedIndex ? 'w-6 bg-rs-white' : 'w-1.5 bg-rs-white/40'
                    }`}
                    aria-label={`Aller au slide ${index + 1}`}
                />
            ))}
        </div>
    )
}