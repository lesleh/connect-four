import { useEffect, useRef, useState } from "react";
import { ROWS, COLS } from "../game/connect4";
import "./Board.css";

interface BoardProps {
  board: Int8Array;
  winningCells: [number, number][] | null;
  lastMove: number | null;
  disabled: boolean;
  onColumnClick: (col: number) => void;
}

interface DroppingPiece {
  row: number;
  col: number;
  key: number;
}

export function Board({ board, winningCells, lastMove, disabled, onColumnClick }: BoardProps) {
  const winSet = new Set(winningCells?.map(([r, c]) => `${r},${c}`) ?? []);
  const [dropping, setDropping] = useState<DroppingPiece | null>(null);
  const prevBoardRef = useRef<Int8Array>(new Int8Array(ROWS * COLS));
  const dropKeyRef = useRef(0);

  useEffect(() => {
    const prev = prevBoardRef.current;
    // Find the newly placed piece
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const idx = r * COLS + c;
        if (board[idx] !== 0 && prev[idx] === 0) {
          dropKeyRef.current++;
          setDropping({ row: r, col: c, key: dropKeyRef.current });
          const timer = setTimeout(() => setDropping(null), 350);
          prevBoardRef.current = new Int8Array(board);
          return () => clearTimeout(timer);
        }
      }
    }
    prevBoardRef.current = new Int8Array(board);
  }, [board]);

  return (
    <div className="board">
      <div className="grid">
        {Array.from({ length: ROWS }, (_, r) =>
          Array.from({ length: COLS }, (_, c) => {
            const val = board[r * COLS + c];
            const isWinning = winSet.has(`${r},${c}`);
            const isLastMove = lastMove === c && val !== 0 &&
              (r === ROWS - 1 || board[(r + 1) * COLS + c] !== 0);
            const canClick = !disabled && board[c] === 0;
            const isDropping = dropping && dropping.row === r && dropping.col === c;
            const dropDistance = isDropping ? r + 1 : 0;
            return (
              <div
                key={`${r}-${c}`}
                className={`cell ${canClick ? "clickable" : ""}`}
                onClick={() => canClick && onColumnClick(c)}
              >
                <div
                  className={[
                    "piece",
                    val === 1 ? "player1" : val === -1 ? "player2" : "empty",
                    isWinning ? "winning" : "",
                    isLastMove ? "last-move" : "",
                    isDropping ? "dropping" : "",
                  ].join(" ")}
                  style={isDropping ? {
                    "--drop-distance": `${-dropDistance * 68}px`,
                  } as React.CSSProperties : undefined}
                />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
