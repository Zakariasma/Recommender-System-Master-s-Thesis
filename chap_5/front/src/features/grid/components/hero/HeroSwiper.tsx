import { useCallback, useEffect, useState } from 'react'
import useEmblaCarousel from 'embla-carousel-react'
import Autoplay from 'embla-carousel-autoplay'
import { HeroCard } from './HeroCard'
import { SwiperDots } from './SwiperDots'
import type { HeroSlide } from '../../types/HeroSlide.ts'

type HeroSwiperProps = {
    slides: HeroSlide[]
}

export function HeroSwiper({ slides }: HeroSwiperProps) {
    const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true }, [
        Autoplay({ delay: 5000, stopOnInteraction: false }),
    ])

    const [selectedIndex, setSelectedIndex] = useState(0)

    const onSelect = useCallback(() => {
        if (!emblaApi) return
        setSelectedIndex(emblaApi.selectedScrollSnap())
    }, [emblaApi])

    useEffect(() => {
        if (!emblaApi) return
        emblaApi.on('select', onSelect)
        onSelect()
    }, [emblaApi, onSelect])

    const scrollTo = (index: number) => emblaApi?.scrollTo(index)

    return (
        <div className="relative flex w-full mb-8 mt-24 justify-center items-center h-[55vh] overflow-hidden">
            <div className="w-[95%] h-full overflow-hidden" ref={emblaRef}>
                <div className="flex h-full">
                    {slides.map((slide) => (
                        <HeroCard key={slide.id} slide={slide} />
                    ))}
                </div>
            </div>

            <SwiperDots count={slides.length} selectedIndex={selectedIndex} onSelect={scrollTo} />
        </div>
    )
}