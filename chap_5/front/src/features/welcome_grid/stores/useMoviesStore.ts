import { create } from 'zustand'
import type { MovieRow } from '../types/MovieRow.ts'
import type { HeroSlide } from '../types/HeroSlide.ts'
import type {MovieDetails} from "../../detail/types/MovieDetails.ts";

interface MovieState {
    rows: MovieRow[]
    detailsCache: Record<string, MovieDetails>
    recommendations: HeroSlide[]
    setRows: (rows: MovieRow[]) => void
    setDetails: (details: MovieDetails) => void
    setRecommendations: (recs: HeroSlide[]) => void
}

export const useMoviesStore = create<MovieState>((set) => ({
    rows: [],
    detailsCache: {},
    recommendations: [],
    setRows: (rows) => set({ rows }),
    setDetails: (details) => set((state) => ({
        detailsCache: { ...state.detailsCache, [details.id]: details }
    })),
    setRecommendations: (recs) => set({ recommendations: recs }),
}))