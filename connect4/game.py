"""Connect 4 game engine with numpy board representation."""

import numpy as np

ROWS = 6
COLS = 7
WIN_LENGTH = 4


class Connect4:
    """Connect 4 game state.

    Board is stored as a (6, 7) int8 array: 0=empty, 1=player1, -1=player2.
    Current player is +1 or -1.
    """

    def __init__(self) -> None:
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self.last_move: int | None = None

    def copy(self) -> "Connect4":
        g = Connect4.__new__(Connect4)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.last_move = self.last_move
        return g

    def legal_moves(self) -> list[int]:
        """Return list of columns that are not full."""
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def play(self, col: int) -> None:
        """Drop a piece in the given column. Mutates state."""
        for row in range(ROWS - 1, -1, -1):
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
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                s = b[r, c] + b[r, c + 1] + b[r, c + 2] + b[r, c + 3]
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                s = b[r, c] + b[r + 1, c] + b[r + 2, c] + b[r + 3, c]
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        # Diagonal (down-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                s = b[r, c] + b[r + 1, c + 1] + b[r + 2, c + 2] + b[r + 3, c + 3]
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        # Diagonal (down-left)
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                s = b[r, c] + b[r + 1, c - 1] + b[r + 2, c - 2] + b[r + 3, c - 3]
                if s == 4:
                    return 1
                if s == -4:
                    return -1
        return None

    def encode(self) -> np.ndarray:
        """Encode board as (2, 6, 7) float32 tensor for the neural network.

        Channel 0: current player's pieces
        Channel 1: opponent's pieces
        """
        state = np.zeros((2, ROWS, COLS), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        return state

    def __str__(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        for r in range(ROWS):
            lines.append(" ".join(symbols[int(self.board[r, c])] for c in range(COLS)))
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)
