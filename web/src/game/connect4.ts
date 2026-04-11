export interface GameConfig {
  rows: number;
  cols: number;
  winLength: number;
}

export const PRESETS: Record<string, GameConfig> = {
  "Connect 3 (3x4)": { rows: 3, cols: 4, winLength: 3 },
  "Connect 4 (6x7)": { rows: 6, cols: 7, winLength: 4 },
  "Connect 5 (9x9)": { rows: 9, cols: 9, winLength: 5 },
};

export class Connect4 {
  config: GameConfig;
  board: Int8Array;
  currentPlayer: 1 | -1;
  lastMove: number | null;

  constructor(config: GameConfig = PRESETS["Connect 4 (6x7)"]) {
    this.config = config;
    this.board = new Int8Array(config.rows * config.cols);
    this.currentPlayer = 1;
    this.lastMove = null;
  }

  get(row: number, col: number): number {
    return this.board[row * this.config.cols + col];
  }

  private set(row: number, col: number, value: number): void {
    this.board[row * this.config.cols + col] = value;
  }

  copy(): Connect4 {
    const g = new Connect4(this.config);
    g.board = new Int8Array(this.board);
    g.currentPlayer = this.currentPlayer;
    g.lastMove = this.lastMove;
    return g;
  }

  legalMoves(): number[] {
    const moves: number[] = [];
    for (let c = 0; c < this.config.cols; c++) {
      if (this.get(0, c) === 0) moves.push(c);
    }
    return moves;
  }

  play(col: number): void {
    for (let row = this.config.rows - 1; row >= 0; row--) {
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
    const { rows, cols, winLength: wl } = this.config;
    // Horizontal
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c <= cols - wl; c++) {
        let s = 0;
        for (let i = 0; i < wl; i++) s += this.get(r, c + i);
        if (s === wl) return 1;
        if (s === -wl) return -1;
      }
    }
    // Vertical
    for (let r = 0; r <= rows - wl; r++) {
      for (let c = 0; c < cols; c++) {
        let s = 0;
        for (let i = 0; i < wl; i++) s += this.get(r + i, c);
        if (s === wl) return 1;
        if (s === -wl) return -1;
      }
    }
    // Diagonal down-right
    for (let r = 0; r <= rows - wl; r++) {
      for (let c = 0; c <= cols - wl; c++) {
        let s = 0;
        for (let i = 0; i < wl; i++) s += this.get(r + i, c + i);
        if (s === wl) return 1;
        if (s === -wl) return -1;
      }
    }
    // Diagonal down-left
    for (let r = 0; r <= rows - wl; r++) {
      for (let c = wl - 1; c < cols; c++) {
        let s = 0;
        for (let i = 0; i < wl; i++) s += this.get(r + i, c - i);
        if (s === wl) return 1;
        if (s === -wl) return -1;
      }
    }
    return null;
  }

  winningCells(): [number, number][] | null {
    const { rows, cols, winLength: wl } = this.config;

    const check = (getPos: (i: number) => [number, number]): [number, number][] | null => {
      const positions: [number, number][] = [];
      let s = 0;
      for (let i = 0; i < wl; i++) {
        const pos = getPos(i);
        positions.push(pos);
        s += this.get(pos[0], pos[1]);
      }
      if (s === wl || s === -wl) return positions;
      return null;
    };

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c <= cols - wl; c++) {
        const result = check((i) => [r, c + i]);
        if (result) return result;
      }
    }
    for (let r = 0; r <= rows - wl; r++) {
      for (let c = 0; c < cols; c++) {
        const result = check((i) => [r + i, c]);
        if (result) return result;
      }
    }
    for (let r = 0; r <= rows - wl; r++) {
      for (let c = 0; c <= cols - wl; c++) {
        const result = check((i) => [r + i, c + i]);
        if (result) return result;
      }
    }
    for (let r = 0; r <= rows - wl; r++) {
      for (let c = wl - 1; c < cols; c++) {
        const result = check((i) => [r + i, c - i]);
        if (result) return result;
      }
    }
    return null;
  }

  encode(): Float32Array {
    const { rows, cols } = this.config;
    const state = new Float32Array(2 * rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = this.get(r, c);
        const idx = r * cols + c;
        if (val === this.currentPlayer) state[idx] = 1;
        if (val === -this.currentPlayer) state[rows * cols + idx] = 1;
      }
    }
    return state;
  }
}
