import { useNavigate } from "react-router"
import { ArrowLeft } from "lucide-react"

export function BackButton() {
    const navigate = useNavigate()

    return (
        <button
            onClick={() => navigate(-1)}
            className="mt-8 w-fit flex items-center gap-2 px-6 py-2 bg-rs-border rounded-md hover:bg-white/20 hover:cursor-pointer transition-all duration-200"
        >
            <ArrowLeft className="w-4 h-4" />
            Retour
        </button>
    )
}