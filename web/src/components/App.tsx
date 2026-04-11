import { useCallback, useEffect, useRef, useState } from "react";
import { Connect4, COLS } from "../game/connect4";
import { mctsSearch } from "../game/mcts";
import { loadModel, isLoaded, predict } from "../ai/model";
import { Board } from "./Board";
import "./App.css";

type GameResult = "win" | "loss" | "draw" | null;

const SIM_OPTIONS = [
  { label: "Beginner", value: 0 },
  { label: "Easy", value: 20 },
  { label: "Medium", value: 50 },
  { label: "Hard", value: 100 },
  { label: "Max", value: 200 },
];

export function App() {
  const [game, setGame] = useState(() => new Connect4());
  const [thinking, setThinking] = useState(false);
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<GameResult>(null);
  const [humanPlayer, setHumanPlayer] = useState<1 | -1>(1);
  const [numSims, setNumSims] = useState(0);
  const [visits, setVisits] = useState<Float32Array | null>(null);
  const gameRef = useRef(game);
  gameRef.current = game;

  useEffect(() => {
    loadModel()
      .then(() => setLoading(false))
      .catch((err) => {
        console.error("Failed to load model:", err);
        setLoading(false);
      });
  }, []);

  const checkResult = useCallback(
    (g: Connect4): GameResult => {
      if (!g.isTerminal()) return null;
      const w = g.winner();
      if (w === null) return "draw";
      return w === humanPlayer ? "win" : "loss";
    },
    [humanPlayer]
  );

  const aiMove = useCallback(
    async (g: Connect4) => {
      setThinking(true);
      // Yield a frame so the UI updates before search blocks
      await new Promise((r) => setTimeout(r, 10));

      let bestCol: number;
      if (numSims === 0) {
        // Beginner: raw network policy, no search
        const { policy } = await predict(g.encode());
        const legal = g.legalMoves();
        // Mask illegal moves
        const masked = new Float32Array(COLS);
        for (const c of legal) masked[c] = policy[c];
        setVisits(masked);
        bestCol = legal[0];
        let bestProb = 0;
        for (const c of legal) {
          if (masked[c] > bestProb) {
            bestProb = masked[c];
            bestCol = c;
          }
        }
      } else {
        const { visits: v } = await mctsSearch(g, numSims);
        setVisits(v);
        bestCol = 0;
        let bestVisits = 0;
        for (let c = 0; c < COLS; c++) {
          if (v[c] > bestVisits) {
            bestVisits = v[c];
            bestCol = c;
          }
        }
      }

      const next = g.copy();
      next.play(bestCol);
      setGame(next);
      setThinking(false);

      const r = checkResult(next);
      if (r) setResult(r);
      return next;
    },
    [numSims, checkResult]
  );

  // AI moves first if human is player 2
  useEffect(() => {
    if (!loading && isLoaded() && game.currentPlayer !== humanPlayer && !result && !thinking) {
      aiMove(game);
    }
  }, [loading, game, humanPlayer, result, thinking, aiMove]);

  const handleColumnClick = useCallback(
    async (col: number) => {
      if (thinking || result || loading) return;
      if (game.currentPlayer !== humanPlayer) return;
      if (game.board[col] !== 0) return;

      const next = game.copy();
      next.play(col);
      setGame(next);
      setVisits(null);

      const r = checkResult(next);
      if (r) {
        setResult(r);
        return;
      }

      aiMove(next);
    },
    [game, humanPlayer, thinking, result, loading, checkResult, aiMove]
  );

  const newGame = useCallback(
    (player: 1 | -1) => {
      setHumanPlayer(player);
      setGame(new Connect4());
      setResult(null);
      setVisits(null);
      setThinking(false);
    },
    []
  );

  const winningCells = game.winningCells();
  const totalVisits = visits ? Array.from(visits).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="app">
      <h1>Connect 4</h1>

      {loading ? (
        <div className="status">Loading AI model...</div>
      ) : (
        <>
          <div className="status">
            {result === "win" && "You win!"}
            {result === "loss" && "AI wins!"}
            {result === "draw" && "Draw!"}
            {!result && thinking && "AI is thinking..."}
            {!result && !thinking && `Your turn (${humanPlayer === 1 ? "Red" : "Yellow"})`}
          </div>

          <Board
            board={game.board}
            winningCells={winningCells}
            lastMove={game.lastMove}
            disabled={thinking || result !== null || game.currentPlayer !== humanPlayer}
            onColumnClick={handleColumnClick}
          />

          {visits && totalVisits > 0 && (
            <div className="visits">
              {Array.from(visits).map((v, i) => (
                <div key={i} className="visit-bar-container">
                  <div
                    className="visit-bar"
                    style={{ height: `${(v / totalVisits) * 100}%` }}
                  />
                  <span className="visit-label">
                    {v > 0 ? Math.round((v / totalVisits) * 100) : ""}
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="controls">
            <div className="difficulty">
              <label>Difficulty:</label>
              {SIM_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  className={numSims === opt.value ? "active" : ""}
                  onClick={() => setNumSims(opt.value)}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            <div className="new-game">
              <button onClick={() => newGame(1)}>New Game (Play Red)</button>
              <button onClick={() => newGame(-1)}>New Game (Play Yellow)</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
