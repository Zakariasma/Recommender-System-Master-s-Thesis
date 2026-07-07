import { Info } from 'lucide-react'

type MoreInfoButtonProps = {
    onClick?: () => void
}

export function MoreInfoButton({ onClick }: MoreInfoButtonProps) {
    return (
        <button
            onClick={onClick}
            className="flex justify-center items-center mt-8 w-56 h-16 border-rs-border bg-rs-white/50 rounded-xl"
        >
            <Info className="w-8 h-8 text-rs-white mr-2" />
            <p className="font-semibold">En savoir plus</p>
        </button>
    )
}