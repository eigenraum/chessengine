// Chess piece graphics: own geometric design, public domain (DESIGN-VISU.md §2).
// Each piece is an SVG fragment on a 45x45 viewBox. Filled shapes inherit the
// piece color from CSS (.piece.white / .piece.black set fill + stroke); the
// stroke is the contrasting color, so bare <line>/<path fill="none"> elements
// double as detail lines.

const BASE = '<rect x="10.5" y="35.5" width="24" height="4" rx="1.6"/>';

export const PIECE_SVG = {
  p: `
    <circle cx="22.5" cy="12" r="5.6"/>
    <path d="M18.6 16.5 h7.8 L25 22.8 c3.4 2.6 4.8 6.3 4.8 11.2 H15.2 c0-4.9 1.4-8.6 4.8-11.2 Z"/>
    ${BASE}`,
  r: `
    <path d="M12.5 10.5 h4.3 v3.6 h3.9 v-3.6 h3.6 v3.6 h3.9 v-3.6 h4.3 v6.8 l-2.6 2.7 v10.6 l2.6 2.7 v2.2 h-20 v-2.2 l2.6-2.7 V20 l-2.6-2.7 Z"/>
    ${BASE}`,
  n: `
    <path d="M13.5 35.5 c0-5.5 1.5-9.9 5-13 c-3-1-5.8-3.7-6.3-7 l3.8-0.5 0.7-3.5 c2.4 0.9 4.2 2 5.5 3.4 l0.8-5.4 3.4 4.6 c6 2.4 9.1 7.6 9.1 14.4 v7 Z"/>
    <circle cx="17.6" cy="16.4" r="0.9" fill="none"/>
    ${BASE}`,
  b: `
    <circle cx="22.5" cy="8.6" r="2.7"/>
    <path d="M22.5 12.6 c5.1 3.6 7.7 7.8 7.7 12.7 c0 3.6-1.9 6.3-4.6 7.2 h-6.2 c-2.7-0.9-4.6-3.6-4.6-7.2 c0-4.9 2.6-9.1 7.7-12.7 Z"/>
    <line x1="20.2" y1="19.5" x2="25.6" y2="24.9" fill="none"/>
    ${BASE}`,
  q: `
    <path d="M11 13.5 l3.6 14 h15.8 l3.6-14 l-5.4 5.6 l-3.3-7.1 l-2.8 7.1 l-2.8-7.1 l-3.3 7.1 Z"/>
    <path d="M14.6 27.5 c-1 3.3-0.2 5.5 1.6 7 h12.6 c1.8-1.5 2.6-3.7 1.6-7 Z"/>
    ${BASE}`,
  k: `
    <path d="M21.3 4.5 h2.4 v2.9 h2.9 v2.4 h-2.9 v2.9 h-2.4 v-2.9 h-2.9 V7.4 h2.9 Z"/>
    <path d="M22.5 13.5 c-6.2 0-10.2 3.6-10.2 8.6 c0 3 1.6 5.6 4.1 7.4 h12.2 c2.5-1.8 4.1-4.4 4.1-7.4 c0-5-4-8.6-10.2-8.6 Z"/>
    <line x1="22.5" y1="17" x2="22.5" y2="26.5" fill="none"/>
    <line x1="18" y1="21.5" x2="27" y2="21.5" fill="none"/>
    ${BASE}`,
};

// symbol: FEN letter ("P" white pawn, "n" black knight, ...) -> <g> element
export function pieceElement(symbol) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const white = symbol === symbol.toUpperCase();
  g.setAttribute("class", `piece ${white ? "white" : "black"}`);
  g.innerHTML = PIECE_SVG[symbol.toLowerCase()];
  return g;
}
