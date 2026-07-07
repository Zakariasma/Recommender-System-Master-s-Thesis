type MovieBackgroundProps = {
    src: string
    title: string
}

export function MovieBackground({ src, title }: MovieBackgroundProps) {
    return (
        <div className="absolute inset-0 w-full h-[80vh] overflow-hidden">
            <img
                src={src}
                alt={title}
                className="w-full h-full object-cover object-center opacity-50 blur-sm"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-rs-black to-transparent"></div>
        </div>
    )
}