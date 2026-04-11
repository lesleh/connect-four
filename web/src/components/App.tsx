import { useCallback, useEffect, useRef, useState } from "react";
import { Connect4, PRESETS } from "../game/connect4";
import type { GameConfig } from "../game/connect4";
import { mctsSearch } from "../game/mcts";
import { loadModel, isLoaded, predict } from "../ai/model";
import { Board } from "./Board";
import "./App.css";

type GameResult = "win" | "loss" | "draw" | null;

const SIM_OPTIONS = [
  { label: "Beginner", value: 0 },
  { label: "Easy", value: -1 },
  { label: "Medium", value: 20 },
  { label: "Hard", value: 50 },
];

const PRESET_NAMES = Object.keys(PRESETS);

export function App() {
  const [presetName, setPresetName] = useState(PRESET_NAMES[1]); // Connect 4 default
  const config = PRESETS[presetName];
  const [game, setGame] = useState(() => new Connect4(config));
  const [thinking, setThinking] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [result, setResult] = useState<GameResult>(null);
  const [humanPlayer, setHumanPlayer] = useState<1 | -1>(1);
  const [numSims, setNumSims] = useState(0);
  const [visits, setVisits] = useState<Float32Array | null>(null);
  const gameRef = useRef(game);
  gameRef.current = game;

  const doLoadModel = useCallback(async (cfg: GameConfig) => {
    setLoading(true);
    setLoadError(null);
    try {
      await loadModel(cfg);
      setLoading(false);
    } catch (err) {
      console.error("Failed to load model:", err);
      setLoadError(`No model found for this variant. Export one first.`);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    doLoadModel(config);
  }, [config, doLoadModel]);

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
      await new Promise((r) => setTimeout(r, 10));

      const cols = g.config.cols;
      let bestCol: number;
      if (numSims <= 0) {
        const { policy } = await predict(g.encode(), g.config);
        const legal = g.legalMoves();
        const masked = new Float32Array(cols);
        for (const c of legal) masked[c] = policy[c];
        setVisits(masked);
        // Beginner (0): 40% random moves. Easy (-1): raw policy, no mistakes.
        if (numSims === 0 && Math.random() < 0.4) {
          bestCol = legal[Math.floor(Math.random() * legal.length)];
        } else {
          bestCol = legal[0];
          let bestProb = 0;
          for (const c of legal) {
            if (masked[c] > bestProb) {
              bestProb = masked[c];
              bestCol = c;
            }
          }
        }
      } else {
        const { visits: v } = await mctsSearch(g, numSims);
        setVisits(v);
        bestCol = 0;
        let bestVisits = 0;
        for (let c = 0; c < cols; c++) {
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

  useEffect(() => {
    if (!loading && !loadError && isLoaded() && game.currentPlayer !== humanPlayer && !result && !thinking) {
      aiMove(game);
    }
  }, [loading, loadError, game, humanPlayer, result, thinking, aiMove]);

  const handleColumnClick = useCallback(
    async (col: number) => {
      if (thinking || result || loading || loadError) return;
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
    [game, humanPlayer, thinking, result, loading, loadError, checkResult, aiMove]
  );

  const newGame = useCallback(
    (player: 1 | -1) => {
      setHumanPlayer(player);
      setGame(new Connect4(config));
      setResult(null);
      setVisits(null);
      setThinking(false);
    },
    [config]
  );

  const switchVariant = useCallback(
    (name: string) => {
      setPresetName(name);
      const cfg = PRESETS[name];
      setGame(new Connect4(cfg));
      setResult(null);
      setVisits(null);
      setThinking(false);
      setHumanPlayer(1);
    },
    []
  );

  const winningCells = game.winningCells();
  const totalVisits = visits ? Array.from(visits).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="app">
      <h1>Connect {config.winLength}</h1>

      <div className="variant-selector">
        {PRESET_NAMES.map((name) => (
          <button
            key={name}
            className={presetName === name ? "active" : ""}
            onClick={() => switchVariant(name)}
          >
            {name}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="status">Loading AI model...</div>
      ) : loadError ? (
        <div className="status error">{loadError}</div>
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
            config={config}
            winningCells={winningCells}
            lastMove={game.lastMove}
            disabled={thinking || result !== null || game.currentPlayer !== humanPlayer}
            onColumnClick={handleColumnClick}
          />

          {visits && totalVisits > 0 && (
            <div className="visits" style={{ gridTemplateColumns: `repeat(${config.cols}, 1fr)` }}>
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
