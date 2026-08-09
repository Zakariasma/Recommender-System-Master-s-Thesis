type MovieHeroInfoProps = {
    title: string
    description: string
}

export function MovieHeroInfo({ title, description }: MovieHeroInfoProps) {
    return (
        <>
            <h1 className="text-4xl mt-4 font-bold text-rs-white">{title}</h1>
            <p className="mt-2 text-gray-300">{description}</p>
        </>
    )
}