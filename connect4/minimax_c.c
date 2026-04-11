/*
 * Fast minimax with alpha-beta pruning for Connect N.
 * Compiled as shared library, called from Python via ctypes.
 *
 * Board is a flat int8 array of rows*cols, row-major.
 * Players are +1 and -1, empty is 0.
 */

#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MAX_COLS 20
#define MAX_ROWS 20
#define MAX_BOARD (MAX_ROWS * MAX_COLS)

typedef struct {
    int8_t board[MAX_BOARD];
    int rows;
    int cols;
    int win_length;
    int current_player;
} Game;

static inline int8_t get(const Game *g, int r, int c) {
    return g->board[r * g->cols + c];
}

static inline void set(Game *g, int r, int c, int8_t val) {
    g->board[r * g->cols + c] = val;
}

static int winner(const Game *g) {
    int rows = g->rows, cols = g->cols, wl = g->win_length;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c <= cols - wl; c++) {
            int s = 0;
            for (int i = 0; i < wl; i++) s += get(g, r, c + i);
            if (s == wl) return 1;
            if (s == -wl) return -1;
        }
    }
    for (int r = 0; r <= rows - wl; r++) {
        for (int c = 0; c < cols; c++) {
            int s = 0;
            for (int i = 0; i < wl; i++) s += get(g, r + i, c);
            if (s == wl) return 1;
            if (s == -wl) return -1;
        }
    }
    for (int r = 0; r <= rows - wl; r++) {
        for (int c = 0; c <= cols - wl; c++) {
            int s = 0;
            for (int i = 0; i < wl; i++) s += get(g, r + i, c + i);
            if (s == wl) return 1;
            if (s == -wl) return -1;
        }
    }
    for (int r = 0; r <= rows - wl; r++) {
        for (int c = wl - 1; c < cols; c++) {
            int s = 0;
            for (int i = 0; i < wl; i++) s += get(g, r + i, c - i);
            if (s == wl) return 1;
            if (s == -wl) return -1;
        }
    }
    return 0; /* no winner */
}

static int legal_moves(const Game *g, int *moves) {
    int n = 0;
    int center = g->cols / 2;
    /* Center-first ordering for better pruning */
    for (int dist = 0; dist < g->cols; dist++) {
        int c1 = center + dist;
        int c2 = center - dist;
        if (c1 < g->cols && get(g, 0, c1) == 0) moves[n++] = c1;
        if (dist > 0 && c2 >= 0 && get(g, 0, c2) == 0) moves[n++] = c2;
    }
    return n;
}

static int is_full(const Game *g) {
    for (int c = 0; c < g->cols; c++) {
        if (get(g, 0, c) == 0) return 0;
    }
    return 1;
}

static void play(Game *g, int col) {
    for (int r = g->rows - 1; r >= 0; r--) {
        if (get(g, r, col) == 0) {
            set(g, r, col, g->current_player);
            g->current_player = -g->current_player;
            return;
        }
    }
}

static int score_window(const int8_t *window, int wl, int player, int opp) {
    int p = 0, o = 0, e = 0;
    for (int i = 0; i < wl; i++) {
        if (window[i] == player) p++;
        else if (window[i] == opp) o++;
        else e++;
    }
    if (p == wl) return 100;
    if (p == wl - 1 && e == 1) return 5;
    if (p == wl - 2 && e == 2) return 2;
    if (o == wl - 1 && e == 1) return -4;
    return 0;
}

static int score_position(const Game *g, int player) {
    int score = 0;
    int rows = g->rows, cols = g->cols, wl = g->win_length;
    int opp = -player;
    int center = cols / 2;
    int8_t window[MAX_COLS];

    /* Center column preference */
    for (int r = 0; r < rows; r++) {
        if (get(g, r, center) == player) score += 3;
    }

    /* Horizontal */
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c <= cols - wl; c++) {
            for (int i = 0; i < wl; i++) window[i] = get(g, r, c + i);
            score += score_window(window, wl, player, opp);
        }
    }
    /* Vertical */
    for (int r = 0; r <= rows - wl; r++) {
        for (int c = 0; c < cols; c++) {
            for (int i = 0; i < wl; i++) window[i] = get(g, r + i, c);
            score += score_window(window, wl, player, opp);
        }
    }
    /* Diagonal down-right */
    for (int r = 0; r <= rows - wl; r++) {
        for (int c = 0; c <= cols - wl; c++) {
            for (int i = 0; i < wl; i++) window[i] = get(g, r + i, c + i);
            score += score_window(window, wl, player, opp);
        }
    }
    /* Diagonal down-left */
    for (int r = 0; r <= rows - wl; r++) {
        for (int c = wl - 1; c < cols; c++) {
            for (int i = 0; i < wl; i++) window[i] = get(g, r + i, c - i);
            score += score_window(window, wl, player, opp);
        }
    }
    return score;
}

static int minimax(Game *g, int depth, int alpha, int beta, int maximizing, int *best_col) {
    int w = winner(g);
    if (w != 0) {
        *best_col = -1;
        if (maximizing) return -10000 - depth;
        else return 10000 + depth;
    }
    if (is_full(g)) {
        *best_col = -1;
        return 0;
    }
    if (depth == 0) {
        *best_col = -1;
        int player = maximizing ? g->current_player : -g->current_player;
        return score_position(g, player);
    }

    int moves[MAX_COLS];
    int n_moves = legal_moves(g, moves);
    *best_col = moves[0];

    if (maximizing) {
        int max_eval = -1000000;
        for (int i = 0; i < n_moves; i++) {
            Game child;
            memcpy(&child, g, sizeof(Game));
            play(&child, moves[i]);
            int dummy;
            int eval = minimax(&child, depth - 1, alpha, beta, 0, &dummy);
            if (eval > max_eval) {
                max_eval = eval;
                *best_col = moves[i];
            }
            if (eval > alpha) alpha = eval;
            if (beta <= alpha) break;
        }
        return max_eval;
    } else {
        int min_eval = 1000000;
        for (int i = 0; i < n_moves; i++) {
            Game child;
            memcpy(&child, g, sizeof(Game));
            play(&child, moves[i]);
            int dummy;
            int eval = minimax(&child, depth - 1, alpha, beta, 1, &dummy);
            if (eval < min_eval) {
                min_eval = eval;
                *best_col = moves[i];
            }
            if (eval < beta) beta = eval;
            if (beta <= alpha) break;
        }
        return min_eval;
    }
}

/*
 * Public API: returns best column for current player.
 * board: flat int8 array, row-major, rows*cols elements
 * current_player: +1 or -1
 */
int minimax_move(const int8_t *board, int rows, int cols, int win_length,
                 int current_player, int depth) {
    Game g;
    g.rows = rows;
    g.cols = cols;
    g.win_length = win_length;
    g.current_player = current_player;
    memcpy(g.board, board, rows * cols * sizeof(int8_t));

    int best_col;
    minimax(&g, depth, -1000000, 1000000, 1, &best_col);
    return best_col;
}
