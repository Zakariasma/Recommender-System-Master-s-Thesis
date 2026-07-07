import {GOLDEN_RATIO} from "../../shared/utils/layout.ts";

const ROW_TITLE_HEIGHT = 40
const ROW_VERTICAL_MARGIN = 24

export function getCardDimensions(rowHeight: number) {
    const cardHeight = Math.max(rowHeight - ROW_TITLE_HEIGHT - ROW_VERTICAL_MARGIN, 80)
    const cardWidth = Math.round(cardHeight / GOLDEN_RATIO)
    return { cardWidth, cardHeight: Math.round(cardHeight) }
}