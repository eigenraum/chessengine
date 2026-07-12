#include "movegen.h"

#include <bit>

#include "attacks.h"

namespace core {

namespace {

int pop_lsb(Bitboard& bb) {
    int sq = std::countr_zero(bb);
    bb &= bb - 1;
    return sq;
}

// Pushes the move, fanned out into all four promotions if a pawn reaches the
// last rank.
void push_pawn_move(MoveList& out, int from, int to) {
    if (rank_of(to) == 0 || rank_of(to) == 7) {
        for (PieceType promotion : {QUEEN, ROOK, BISHOP, KNIGHT})
            out.push(Move(from, to, promotion));
    } else {
        out.push(Move(from, to));
    }
}

void generate_pawn_moves(const Board& b, MoveList& out) {
    const Color us = b.side_to_move();
    const int push = us == WHITE ? 8 : -8;
    const int start_rank = us == WHITE ? 1 : 6;
    const Bitboard occupied = b.occupied();

    Bitboard pawns = b.pieces(us, PAWN);
    while (pawns) {
        int from = pop_lsb(pawns);

        int to = from + push;
        if (!(occupied & square_bb(to))) {
            push_pawn_move(out, from, to);
            if (rank_of(from) == start_rank && !(occupied & square_bb(to + push)))
                out.push(Move(from, to + push));  // double push
        }

        Bitboard captures = PAWN_ATTACKS[us][from] & b.pieces(opponent(us));
        while (captures) push_pawn_move(out, from, pop_lsb(captures));

        if (b.ep_square() >= 0 && (PAWN_ATTACKS[us][from] & square_bb(b.ep_square())))
            out.push(Move(from, b.ep_square()));
    }
}

void generate_piece_moves(const Board& b, MoveList& out) {
    const Color us = b.side_to_move();
    const Bitboard occupied = b.occupied();
    const Bitboard own = b.pieces(us);

    for (PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
        Bitboard pieces = b.pieces(us, pt);
        while (pieces) {
            int from = pop_lsb(pieces);
            Bitboard attacks = pt == KNIGHT ? KNIGHT_ATTACKS[from]
                             : pt == BISHOP ? bishop_attacks(from, occupied)
                             : pt == ROOK   ? rook_attacks(from, occupied)
                             : pt == QUEEN  ? queen_attacks(from, occupied)
                                            : KING_ATTACKS[from];
            Bitboard targets = attacks & ~own;
            while (targets) out.push(Move(from, pop_lsb(targets)));
        }
    }
}

void generate_castling(const Board& b, MoveList& out) {
    const Color us = b.side_to_move();
    const Color them = opponent(us);
    const int king = us == WHITE ? 4 : 60;  // e1 / e8 (guaranteed by castling rights)
    if (!(b.can_castle(us, true) || b.can_castle(us, false))) return;
    if (b.is_attacked(king, them)) return;  // no castling out of check

    const Bitboard occupied = b.occupied();
    if (b.can_castle(us, true)  // O-O: f and g empty, king's path not attacked
        && !(occupied & (square_bb(king + 1) | square_bb(king + 2)))
        && !b.is_attacked(king + 1, them) && !b.is_attacked(king + 2, them))
        out.push(Move(king, king + 2));
    if (b.can_castle(us, false)  // O-O-O: b, c and d empty, king's path not attacked
        && !(occupied & (square_bb(king - 1) | square_bb(king - 2) | square_bb(king - 3)))
        && !b.is_attacked(king - 1, them) && !b.is_attacked(king - 2, them))
        out.push(Move(king, king - 2));
}

}  // namespace

MoveList generate_legal(const Board& board) {
    MoveList pseudo;
    generate_pawn_moves(board, pseudo);
    generate_piece_moves(board, pseudo);
    generate_castling(board, pseudo);

    const Color us = board.side_to_move();
    MoveList legal;
    for (Move m : pseudo) {
        Board next = board;  // copy-make legality check
        next.apply(m);
        if (!next.is_attacked(next.king_square(us), opponent(us))) legal.push(m);
    }
    return legal;
}

uint64_t perft(const Board& board, int depth) {
    if (depth == 0) return 1;
    MoveList moves = generate_legal(board);
    if (depth == 1) return uint64_t(moves.size());
    uint64_t nodes = 0;
    for (Move m : moves) {
        Board next = board;
        next.apply(m);
        nodes += perft(next, depth - 1);
    }
    return nodes;
}

}  // namespace core
