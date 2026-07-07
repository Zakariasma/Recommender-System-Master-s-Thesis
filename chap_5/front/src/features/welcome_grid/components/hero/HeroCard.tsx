import { MovieBadges } from './MovieBadges'
import { MovieHeroInfo } from './MovieHeroInfo'
import { MoreInfoButton } from './MoreInfoButton'
import type {HeroSlide} from "../../types/HeroSlide.ts";

type HeroCardProps = {
    slide: HeroSlide
}

export function HeroCard({ slide }: HeroCardProps) {
    return (
        <div className="relative flex w-full flex-[0_0_100%] h-full border-2 border-rs-border rounded-4xl overflow-hidden">
            <img
                src={slide.background}
                alt={slide.title}
                className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-r from-rs-black via-rs-black/60 to-transparent" />

            <div className="relative flex w-2/5 flex-col h-full ml-8 justify-center z-10">
                <MovieBadges tag={slide.tag} year={slide.year} rating={slide.rating} />
                <MovieHeroInfo title={slide.title} description={slide.description} />
                <MoreInfoButton />
            </div>
        </div>
    )
}