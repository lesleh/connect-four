"""Minimax opponent with alpha-beta pruning for evaluation."""

import numpy as np

from .game import COLS, ROWS, Connect4


def _score_position(board: np.ndarray, player: int) -> int:
    """Heuristic score for non-terminal positions."""
    score = 0
    opp = -player

    # Prefer center column
    center_col = board[:, 3]
    score += int(np.sum(center_col == player)) * 3

    # Score all windows of 4
    for r in range(ROWS):
        for c in range(COLS - 3):
            window = [int(board[r, c + i]) for i in range(4)]
            score += _score_window(window, player, opp)
    for r in range(ROWS - 3):
        for c in range(COLS):
            window = [int(board[r + i, c]) for i in range(4)]
            score += _score_window(window, player, opp)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [int(board[r + i, c + i]) for i in range(4)]
            score += _score_window(window, player, opp)
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            window = [int(board[r + i, c - i]) for i in range(4)]
            score += _score_window(window, player, opp)

    return score


def _score_window(window: list[int], player: int, opp: int) -> int:
    p_count = window.count(player)
    o_count = window.count(opp)
    empty = window.count(0)

    if p_count == 4:
        return 100
    if p_count == 3 and empty == 1:
        return 5
    if p_count == 2 and empty == 2:
        return 2
    if o_count == 3 and empty == 1:
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
        # If there's a winner, the last player to move won.
        # When maximizing, current_player hasn't moved yet.
        if maximizing:
            return None, -10000 - depth  # opponent just won
        else:
            return None, 10000 + depth   # we just won

    if depth == 0:
        player = game.current_player if maximizing else -game.current_player
        return None, _score_position(game.board, player)

    legal = game.legal_moves()
    # Check center-ish columns first for better pruning
    legal.sort(key=lambda c: abs(c - 3))

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
