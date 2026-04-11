"""Minimax opponent with alpha-beta pruning for evaluation."""

import numpy as np

from .game import Connect4


def _score_position(board: np.ndarray, player: int, rows: int, cols: int, wl: int) -> int:
    """Heuristic score for non-terminal positions."""
    score = 0
    opp = -player

    # Prefer center column
    center = cols // 2
    center_col = board[:, center]
    score += int(np.sum(center_col == player)) * 3

    # Score all windows
    for r in range(rows):
        for c in range(cols - wl + 1):
            window = [int(board[r, c + i]) for i in range(wl)]
            score += _score_window(window, player, opp, wl)
    for r in range(rows - wl + 1):
        for c in range(cols):
            window = [int(board[r + i, c]) for i in range(wl)]
            score += _score_window(window, player, opp, wl)
    for r in range(rows - wl + 1):
        for c in range(cols - wl + 1):
            window = [int(board[r + i, c + i]) for i in range(wl)]
            score += _score_window(window, player, opp, wl)
    for r in range(rows - wl + 1):
        for c in range(wl - 1, cols):
            window = [int(board[r + i, c - i]) for i in range(wl)]
            score += _score_window(window, player, opp, wl)

    return score


def _score_window(window: list[int], player: int, opp: int, wl: int) -> int:
    p_count = window.count(player)
    o_count = window.count(opp)
    empty = window.count(0)

    if p_count == wl:
        return 100
    if p_count == wl - 1 and empty == 1:
        return 5
    if p_count == wl - 2 and empty == 2:
        return 2
    if o_count == wl - 1 and empty == 1:
        return -4
    return 0


def minimax(
    game: Connect4,
    depth: int,
    alpha: float = -float("inf"),
    beta: float = float("inf"),
    maximizing: bool = True,
) -> tuple[int | None, float]:
    """Minimax with alpha-beta pruning.

    Returns (best_column, score) from the perspective of game.current_player
    when maximizing=True.
    """
    if game.is_terminal():
        winner = game.winner()
        if winner is None:
            return None, 0
        if maximizing:
            return None, -10000 - depth
        else:
            return None, 10000 + depth

    cfg = game.config
    if depth == 0:
        player = game.current_player if maximizing else -game.current_player
        return None, _score_position(game.board, player, cfg.rows, cfg.cols, cfg.win_length)

    legal = game.legal_moves()
    center = cfg.cols // 2
    legal.sort(key=lambda c: abs(c - center))

    best_col = legal[0]

    if maximizing:
        max_eval = -float("inf")
        for col in legal:
            child = game.copy()
            child.play(col)
            _, eval_score = minimax(child, depth - 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_col = col
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return best_col, max_eval
    else:
        min_eval = float("inf")
        for col in legal:
            child = game.copy()
            child.play(col)
            _, eval_score = minimax(child, depth - 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_col = col
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return best_col, min_eval


def minimax_move(game: Connect4, depth: int = 5) -> int:
    """Return the best move for the current player using minimax."""
    col, _ = minimax(game, depth, maximizing=True)
    return col
