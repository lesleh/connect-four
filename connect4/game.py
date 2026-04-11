"""Configurable Connect N game engine with numpy board representation."""

import dataclasses

import numpy as np

ROWS = 6
COLS = 7
WIN_LENGTH = 4


@dataclasses.dataclass(frozen=True)
class GameConfig:
    rows: int = 6
    cols: int = 7
    win_length: int = 4


DEFAULT_CONFIG = GameConfig()


class Connect4:
    """Connect N game state.

    Board is stored as a (rows, cols) int8 array: 0=empty, 1=player1, -1=player2.
    Current player is +1 or -1.
    """

    def __init__(self, config: GameConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.board = np.zeros((config.rows, config.cols), dtype=np.int8)
        self.current_player = 1
        self.last_move: int | None = None

    def copy(self) -> "Connect4":
        g = Connect4.__new__(Connect4)
        g.config = self.config
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.last_move = self.last_move
        return g

    def legal_moves(self) -> list[int]:
        """Return list of columns that are not full."""
        return [c for c in range(self.config.cols) if self.board[0, c] == 0]

    def play(self, col: int) -> None:
        """Drop a piece in the given column. Mutates state."""
        for row in range(self.config.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.last_move = col
                self.current_player *= -1
                return
        raise ValueError(f"Column {col} is full")

    def is_terminal(self) -> bool:
        """Return True if the game is over (win or draw)."""
        return self.winner() is not None or len(self.legal_moves()) == 0

    def winner(self) -> int | None:
        """Return 1, -1, or None. Does not detect draws (use is_terminal)."""
        b = self.board
        rows, cols, wl = self.config.rows, self.config.cols, self.config.win_length
        target = wl

        # Horizontal
        for r in range(rows):
            for c in range(cols - wl + 1):
                s = sum(int(b[r, c + i]) for i in range(wl))
                if s == target:
                    return 1
                if s == -target:
                    return -1
        # Vertical
        for r in range(rows - wl + 1):
            for c in range(cols):
                s = sum(int(b[r + i, c]) for i in range(wl))
                if s == target:
                    return 1
                if s == -target:
                    return -1
        # Diagonal (down-right)
        for r in range(rows - wl + 1):
            for c in range(cols - wl + 1):
                s = sum(int(b[r + i, c + i]) for i in range(wl))
                if s == target:
                    return 1
                if s == -target:
                    return -1
        # Diagonal (down-left)
        for r in range(rows - wl + 1):
            for c in range(wl - 1, cols):
                s = sum(int(b[r + i, c - i]) for i in range(wl))
                if s == target:
                    return 1
                if s == -target:
                    return -1
        return None

    def encode(self) -> np.ndarray:
        """Encode board as (2, rows, cols) float32 tensor for the neural network.

        Channel 0: current player's pieces
        Channel 1: opponent's pieces
        """
        rows, cols = self.config.rows, self.config.cols
        state = np.zeros((2, rows, cols), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        return state

    def __str__(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        for r in range(self.config.rows):
            lines.append(" ".join(symbols[int(self.board[r, c])] for c in range(self.config.cols)))
        lines.append(" ".join(str(c) for c in range(self.config.cols)))
        return "\n".join(lines)
