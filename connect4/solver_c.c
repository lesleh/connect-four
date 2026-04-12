/*
 * Perfect Connect 4 solver using alpha-beta with bitboard representation
 * and transposition table. Searches to terminal — no heuristics.
 *
 * Bitboard layout: each column uses 7 bits (6 rows + 1 top sentinel).
 * Bit index = col * 7 + row, where row 0 = bottom.
 *
 * Called from Python via ctypes.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define ROWS 6
#define COLS 7
#define BOARD_SIZE (ROWS * COLS)

/* Transposition table */
#define TT_SIZE (1 << 23)  /* ~8M entries, ~128MB */
#define TT_MASK (TT_SIZE - 1)

typedef struct {
    uint64_t key;
    int8_t score;
    uint8_t flag;  /* 0=empty, 1=exact, 2=lower, 3=upper */
} TTEntry;

static TTEntry tt[TT_SIZE];
static int nodes_searched;
static int node_limit;
static int search_aborted;

static inline int popcount(uint64_t x) {
    return __builtin_popcountll(x);
}

/* Check if a bitboard has 4 in a row */
static inline int has_won(uint64_t bb) {
    uint64_t y;
    /* Horizontal: shift by 7 (column width including sentinel) */
    y = bb & (bb >> 7);
    if (y & (y >> 14)) return 1;
    /* Vertical: shift by 1 */
    y = bb & (bb >> 1);
    if (y & (y >> 2)) return 1;
    /* Diagonal /: shift by 6 */
    y = bb & (bb >> 6);
    if (y & (y >> 12)) return 1;
    /* Diagonal \: shift by 8 */
    y = bb & (bb >> 8);
    if (y & (y >> 16)) return 1;
    return 0;
}

typedef struct {
    uint64_t position;  /* current player's pieces */
    uint64_t mask;      /* all pieces */
    int moves;          /* number of moves played */
} Bitboard;

static inline uint64_t bottom_mask(void) {
    uint64_t m = 0;
    for (int c = 0; c < COLS; c++)
        m |= (uint64_t)1 << (c * 7);
    return m;
}

static inline uint64_t board_mask(void) {
    uint64_t m = 0;
    for (int c = 0; c < COLS; c++)
        m |= ((uint64_t)0x3F << (c * 7)); /* 6 bits per column */
    return m;
}

static inline uint64_t top_mask(int col) {
    return (uint64_t)1 << (col * 7 + ROWS - 1);
}

static inline uint64_t bottom_mask_col(int col) {
    return (uint64_t)1 << (col * 7);
}

static inline uint64_t column_mask(int col) {
    return ((uint64_t)0x3F) << (col * 7);
}

static inline int can_play(const Bitboard *b, int col) {
    return (b->mask & top_mask(col)) == 0;
}

static inline void play(Bitboard *b, int col) {
    uint64_t move = (b->mask + bottom_mask_col(col)) & column_mask(col);
    b->position ^= b->mask;  /* switch perspective */
    b->mask |= move;
    b->moves++;
}

static inline uint64_t key(const Bitboard *b) {
    return b->position + b->mask;
}

static inline uint64_t opponent_position(const Bitboard *b) {
    return b->position ^ b->mask;
}

/* Check if playing a column wins immediately */
static inline int is_winning_move(const Bitboard *b, int col) {
    uint64_t pos = b->position;
    uint64_t move = (b->mask + bottom_mask_col(col)) & column_mask(col);
    return has_won(pos | move);
}

/* Score: positive = current player wins, 0 = draw, negative = loses.
 * Magnitude indicates how soon: (BOARD_SIZE+1-moves)/2 is max.
 * A score of +1 means current player wins on the last possible move.
 */

static int negamax(Bitboard *b, int alpha, int beta) {
    nodes_searched++;

    if (node_limit > 0 && nodes_searched >= node_limit) {
        search_aborted = 1;
        return 0;
    }

    /* Draw check */
    if (b->moves >= BOARD_SIZE) return 0;

    /* Check if current player can win immediately */
    for (int c = 0; c < COLS; c++) {
        if (can_play(b, c) && is_winning_move(b, c)) {
            return (BOARD_SIZE + 1 - b->moves) / 2;
        }
    }

    /* Upper bound: best possible score */
    int max_score = (BOARD_SIZE - 1 - b->moves) / 2;
    if (beta > max_score) {
        beta = max_score;
        if (alpha >= beta) return beta;
    }

    /* Transposition table lookup */
    uint64_t k = key(b);
    uint32_t idx = (uint32_t)(k & TT_MASK);
    if (tt[idx].key == k && tt[idx].flag != 0) {
        if (tt[idx].flag == 1) return tt[idx].score;  /* exact */
        if (tt[idx].flag == 2 && tt[idx].score > alpha) {
            alpha = tt[idx].score;
            if (alpha >= beta) return alpha;
        }
        if (tt[idx].flag == 3 && tt[idx].score < beta) {
            beta = tt[idx].score;
            if (alpha >= beta) return beta;
        }
    }

    int best = -100;
    /* Column ordering: center first */
    static const int col_order[COLS] = {3, 2, 4, 1, 5, 0, 6};

    for (int i = 0; i < COLS; i++) {
        int c = col_order[i];
        if (!can_play(b, c)) continue;

        Bitboard child = *b;
        play(&child, c);

        int score = -negamax(&child, -beta, -alpha);

        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    /* Store in TT */
    tt[idx].key = k;
    tt[idx].score = (int8_t)best;
    if (best <= alpha) tt[idx].flag = 3;       /* upper bound */
    else if (best >= beta) tt[idx].flag = 2;   /* lower bound */
    else tt[idx].flag = 1;                      /* exact */

    return best;
}

/* Convert flat int8 board (row-major, row 0 = top) to bitboard */
static void board_to_bitboard(const int8_t *board, int current_player, Bitboard *bb) {
    bb->position = 0;
    bb->mask = 0;
    bb->moves = 0;

    for (int c = 0; c < COLS; c++) {
        for (int r = ROWS - 1; r >= 0; r--) {
            int val = board[r * COLS + c];
            if (val != 0) {
                int bit_row = ROWS - 1 - r;  /* flip: bit row 0 = bottom */
                uint64_t bit = (uint64_t)1 << (c * 7 + bit_row);
                bb->mask |= bit;
                if (val == current_player) {
                    bb->position |= bit;
                }
                bb->moves++;
            }
        }
    }
}

/*
 * Public API: returns the score for each legal column.
 * scores[col] = exact score if legal, -100 if illegal.
 * Returns the best column.
 */
int solver_move(const int8_t *board, int current_player, int *scores, int max_nodes) {
    Bitboard bb;
    board_to_bitboard(board, current_player, &bb);

    /* Clear TT */
    memset(tt, 0, sizeof(tt));
    nodes_searched = 0;
    node_limit = max_nodes;
    search_aborted = 0;

    int best_score = -100;
    int best_col = -1;
    static const int col_order[COLS] = {3, 2, 4, 1, 5, 0, 6};

    for (int i = 0; i < COLS; i++) {
        int c = col_order[i];
        if (!can_play(&bb, c)) {
            scores[c] = -100;
            continue;
        }

        /* Check for immediate win */
        if (is_winning_move(&bb, c)) {
            scores[c] = (BOARD_SIZE + 1 - bb.moves) / 2;
            if (scores[c] > best_score) {
                best_score = scores[c];
                best_col = c;
            }
            continue;
        }

        Bitboard child = bb;
        play(&child, c);

        /* Negamax returns score from child's perspective, negate for ours */
        int score = -negamax(&child, -100, 100);
        scores[c] = score;

        if (score > best_score) {
            best_score = score;
            best_col = c;
        }
    }

    return best_col;
}

int get_nodes_searched(void) {
    return nodes_searched;
}

int was_search_complete(void) {
    return !search_aborted;
}
