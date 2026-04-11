export const ROWS = 6;
export const COLS = 7;

export class Connect4 {
  board: Int8Array;
  currentPlayer: 1 | -1;
  lastMove: number | null;

  constructor() {
    this.board = new Int8Array(ROWS * COLS);
    this.currentPlayer = 1;
    this.lastMove = null;
  }

  private get(row: number, col: number): number {
    return this.board[row * COLS + col];
  }

  private set(row: number, col: number, value: number): void {
    this.board[row * COLS + col] = value;
  }

  copy(): Connect4 {
    const g = new Connect4();
    g.board = new Int8Array(this.board);
    g.currentPlayer = this.currentPlayer;
    g.lastMove = this.lastMove;
    return g;
  }

  legalMoves(): number[] {
    const moves: number[] = [];
    for (let c = 0; c < COLS; c++) {
      if (this.get(0, c) === 0) moves.push(c);
    }
    return moves;
  }

  play(col: number): void {
    for (let row = ROWS - 1; row >= 0; row--) {
      if (this.get(row, col) === 0) {
        this.set(row, col, this.currentPlayer);
        this.lastMove = col;
        this.currentPlayer = (this.currentPlayer === 1 ? -1 : 1) as 1 | -1;
        return;
      }
    }
    throw new Error(`Column ${col} is full`);
  }

  isTerminal(): boolean {
    return this.winner() !== null || this.legalMoves().length === 0;
  }

  winner(): number | null {
    // Horizontal
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS - 3; c++) {
        const s = this.get(r, c) + this.get(r, c + 1) + this.get(r, c + 2) + this.get(r, c + 3);
        if (s === 4) return 1;
        if (s === -4) return -1;
      }
    }
    // Vertical
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 0; c < COLS; c++) {
        const s = this.get(r, c) + this.get(r + 1, c) + this.get(r + 2, c) + this.get(r + 3, c);
        if (s === 4) return 1;
        if (s === -4) return -1;
      }
    }
    // Diagonal down-right
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 0; c < COLS - 3; c++) {
        const s = this.get(r, c) + this.get(r + 1, c + 1) + this.get(r + 2, c + 2) + this.get(r + 3, c + 3);
        if (s === 4) return 1;
        if (s === -4) return -1;
      }
    }
    // Diagonal down-left
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 3; c < COLS; c++) {
        const s = this.get(r, c) + this.get(r + 1, c - 1) + this.get(r + 2, c - 2) + this.get(r + 3, c - 3);
        if (s === 4) return 1;
        if (s === -4) return -1;
      }
    }
    return null;
  }

  winningCells(): [number, number][] | null {
    const check = (positions: [number, number][]): [number, number][] | null => {
      const s = positions.reduce((sum, [r, c]) => sum + this.get(r, c), 0);
      if (s === 4 || s === -4) return positions;
      return null;
    };

    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS - 3; c++) {
        const result = check([[r, c], [r, c + 1], [r, c + 2], [r, c + 3]]);
        if (result) return result;
      }
    }
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 0; c < COLS; c++) {
        const result = check([[r, c], [r + 1, c], [r + 2, c], [r + 3, c]]);
        if (result) return result;
      }
    }
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 0; c < COLS - 3; c++) {
        const result = check([[r, c], [r + 1, c + 1], [r + 2, c + 2], [r + 3, c + 3]]);
        if (result) return result;
      }
    }
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 3; c < COLS; c++) {
        const result = check([[r, c], [r + 1, c - 1], [r + 2, c - 2], [r + 3, c - 3]]);
        if (result) return result;
      }
    }
    return null;
  }

  encode(): Float32Array {
    const state = new Float32Array(2 * ROWS * COLS);
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const val = this.get(r, c);
        const idx = r * COLS + c;
        if (val === this.currentPlayer) state[idx] = 1;
        if (val === -this.currentPlayer) state[ROWS * COLS + idx] = 1;
      }
    }
    return state;
  }
}
