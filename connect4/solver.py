"""Perfect Connect 4 solver using C alpha-beta with bitboard + transposition table."""

import ctypes
from pathlib import Path

from .game import Connect4

_lib = None
_so_path = Path(__file__).parent / "solver_c.so"
if _so_path.exists():
    try:
        _lib = ctypes.CDLL(str(_so_path))
        _lib.solver_move.restype = ctypes.c_int
        _lib.solver_move.argtypes = [
            ctypes.POINTER(ctypes.c_int8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        _lib.get_nodes_searched.restype = ctypes.c_int
        _lib.get_nodes_searched.argtypes = []
        _lib.was_search_complete.restype = ctypes.c_int
        _lib.was_search_complete.argtypes = []
    except OSError:
        _lib = None


def is_available() -> bool:
    return _lib is not None


def solver_move(game: Connect4, max_nodes: int = 50_000_000) -> tuple[int, bool]:
    """Return (best_column, is_exact).

    If max_nodes is hit before solving, returns the best move found so far
    with is_exact=False. Default 50M nodes (~5-10 seconds).
    """
    if _lib is None:
        raise RuntimeError("Solver not available. Compile solver_c.c first.")

    board = game.board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    scores = (ctypes.c_int * 7)()
    best = _lib.solver_move(board, game.current_player, scores, max_nodes)
    exact = bool(_lib.was_search_complete())
    return best, exact


def solver_scores(game: Connect4, max_nodes: int = 50_000_000) -> tuple[list[int | None], bool]:
    """Return (scores_per_column, is_exact). None for illegal columns."""
    if _lib is None:
        raise RuntimeError("Solver not available. Compile solver_c.c first.")

    board = game.board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    scores = (ctypes.c_int * 7)()
    _lib.solver_move(board, game.current_player, scores, max_nodes)
    exact = bool(_lib.was_search_complete())
    return [scores[c] if scores[c] != -100 else None for c in range(7)], exact


def nodes_searched() -> int:
    if _lib is None:
        return 0
    return _lib.get_nodes_searched()
