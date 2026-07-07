import {ChevronRight} from 'lucide-react'

type ScrollRightButtonProps = {
    onClick: () => void
}

export function ScrollRightButton({onClick}: ScrollRightButtonProps) {
    return (
        <button
            onClick={onClick}
            className="absolute right-0 top-1/2 -translate-y-1/2 h-[80%] w-10 flex items-center justify-center backdrop-blur-md bg-white/20 hover:bg-white/30 transition-colors rounded-l-md z-10"
            aria-label="Scroll right"
        >
            <ChevronRight className="text-rs-white w-6 h-6"/>
        </button>
    )
}