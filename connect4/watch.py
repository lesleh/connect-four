"""TUI training mode: watch self-play games live while the model trains."""

import argparse
import curses
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .game import COLS, ROWS, Connect4
from .network import Connect4Net
from .parallel import ParallelSelfPlay
from .train import (
    BATCH_SIZE,
    C_PUCT,
    CHECKPOINT_DIR,
    EPOCHS_PER_ITERATION,
    GAMES_PER_ITERATION,
    LEARNING_RATE,
    NUM_CHANNELS,
    NUM_ITERATIONS,
    NUM_RES_BLOCKS,
    NUM_SIMULATIONS,
    REPLAY_BUFFER_SIZE,
    TEMPERATURE_THRESHOLD,
    WEIGHT_DECAY,
    train_network,
)

NUM_PARALLEL = 16  # games played simultaneously

# Board drawing constants
CELL_W = 4
BOARD_TOP = 5
BOARD_LEFT = 2

# Colors
EMPTY_PAIR = 1
P1_PAIR = 2
P2_PAIR = 3
BORDER_PAIR = 4
HIGHLIGHT_PAIR = 5
BAR_PAIR = 6
HEADER_PAIR = 7
DIM_PAIR = 8


def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(EMPTY_PAIR, curses.COLOR_WHITE, -1)
    curses.init_pair(P1_PAIR, curses.COLOR_RED, -1)
    curses.init_pair(P2_PAIR, curses.COLOR_YELLOW, -1)
    curses.init_pair(BORDER_PAIR, curses.COLOR_BLUE, -1)
    curses.init_pair(HIGHLIGHT_PAIR, curses.COLOR_GREEN, -1)
    curses.init_pair(BAR_PAIR, curses.COLOR_CYAN, -1)
    curses.init_pair(HEADER_PAIR, curses.COLOR_WHITE, -1)
    curses.init_pair(DIM_PAIR, curses.COLOR_WHITE, -1)


def draw_board(win: curses.window, game: Connect4, last_row: int = -1, last_col: int = -1) -> None:
    y0 = BOARD_TOP
    x0 = BOARD_LEFT

    top = "+" + ("---+" * COLS)
    win.addstr(y0, x0, top, curses.color_pair(BORDER_PAIR))

    for r in range(ROWS):
        row_y = y0 + 1 + r * 2
        win.addstr(row_y, x0, "|", curses.color_pair(BORDER_PAIR))
        for c in range(COLS):
            cell_x = x0 + 1 + c * CELL_W
            val = int(game.board[r, c])
            if val == 1:
                attr = curses.color_pair(P1_PAIR) | curses.A_BOLD
                ch = " X "
            elif val == -1:
                attr = curses.color_pair(P2_PAIR) | curses.A_BOLD
                ch = " O "
            else:
                attr = curses.color_pair(EMPTY_PAIR) | curses.A_DIM
                ch = " . "
            if r == last_row and c == last_col:
                attr |= curses.A_REVERSE
            win.addstr(row_y, cell_x, ch, attr)
            win.addstr(row_y, cell_x + 3, "|", curses.color_pair(BORDER_PAIR))
        sep = "+" + ("---+" * COLS)
        win.addstr(row_y + 1, x0, sep, curses.color_pair(BORDER_PAIR))

    num_y = y0 + 1 + ROWS * 2
    nums = "  " + "   ".join(str(c) for c in range(COLS))
    win.addstr(num_y, x0, nums, curses.color_pair(DIM_PAIR) | curses.A_DIM)


def draw_policy_bars(win: curses.window, policy: np.ndarray, game: Connect4) -> None:
    x0 = BOARD_LEFT + CELL_W * COLS + 6
    y0 = BOARD_TOP
    max_bar = 20

    win.addstr(y0, x0, "MCTS Policy", curses.color_pair(HEADER_PAIR) | curses.A_BOLD)
    y0 += 1
    legal = set(game.legal_moves())
    max_p = max(policy) if max(policy) > 0 else 1.0
    for c in range(COLS):
        y = y0 + c
        label = f"Col {c}: "
        win.addstr(y, x0, label, curses.color_pair(DIM_PAIR))
        if c not in legal:
            win.addstr(y, x0 + len(label), "---", curses.color_pair(DIM_PAIR) | curses.A_DIM)
            continue
        bar_len = int(policy[c] / max_p * max_bar) if max_p > 0 else 0
        bar = "#" * bar_len
        pct = f" {policy[c]*100:5.1f}%"
        win.addstr(y, x0 + len(label), bar, curses.color_pair(BAR_PAIR) | curses.A_BOLD)
        win.addstr(y, x0 + len(label) + bar_len, pct, curses.color_pair(DIM_PAIR))


def draw_training_header(win: curses.window, stats: dict) -> None:
    y = 1
    x = BOARD_LEFT

    iteration = stats.get("iteration", 0)
    game_num = stats.get("game_num", 0)
    games_total = stats.get("games_total", GAMES_PER_ITERATION)
    phase = stats.get("phase", "starting")
    p_loss = stats.get("policy_loss")
    v_loss = stats.get("value_loss")
    buffer_size = stats.get("buffer_size", 0)
    wins = stats.get("wins", {})

    # Line 1: iteration + phase
    iter_str = f"Iteration {iteration}"
    win.addstr(y, x, iter_str, curses.color_pair(HEADER_PAIR) | curses.A_BOLD)

    if phase == "self-play":
        phase_str = f"  Self-play: game {game_num}/{games_total} ({NUM_PARALLEL} parallel, fp16)"
        win.addstr(y, x + len(iter_str), phase_str, curses.color_pair(HIGHLIGHT_PAIR))
    elif phase == "training":
        win.addstr(y, x + len(iter_str), "  Training network...", curses.color_pair(BAR_PAIR) | curses.A_BOLD)
    elif phase == "done":
        win.addstr(y, x + len(iter_str), "  Training complete!", curses.color_pair(HIGHLIGHT_PAIR) | curses.A_BOLD)

    # Line 2: stats
    y += 1
    stats_parts = [f"Buffer: {buffer_size}"]
    if p_loss is not None:
        stats_parts.append(f"P-loss: {p_loss:.4f}")
    if v_loss is not None:
        stats_parts.append(f"V-loss: {v_loss:.4f}")
    if wins:
        x_wins = wins.get(1, 0)
        o_wins = wins.get(-1, 0)
        draws = wins.get(0, 0)
        stats_parts.append(f"X:{x_wins} O:{o_wins} D:{draws}")
    win.addstr(y, x, "  ".join(stats_parts), curses.color_pair(DIM_PAIR))

    # Line 3: current game info
    y += 1
    move_num = stats.get("move_num", 0)
    current_player = stats.get("current_player", 1)
    player_name = "X (Red)" if current_player == 1 else "O (Yellow)"
    player_pair = P1_PAIR if current_player == 1 else P2_PAIR
    game_status = stats.get("game_status", "")

    win.addstr(y, x, f"Move {move_num:3d}   ", curses.color_pair(HEADER_PAIR) | curses.A_BOLD)
    win.addstr(y, x + 12, "Turn: ", curses.color_pair(DIM_PAIR))
    win.addstr(y, x + 18, player_name, curses.color_pair(player_pair) | curses.A_BOLD)
    if game_status:
        win.addstr(y, x + 30, game_status, curses.color_pair(HIGHLIGHT_PAIR) | curses.A_BOLD)


def draw_footer(win: curses.window, max_y: int) -> None:
    y = max_y - 1
    win.addstr(y, BOARD_LEFT, "[Q] Quit  [+/-] Speed  [Space] Pause",
               curses.color_pair(DIM_PAIR) | curses.A_DIM)


def find_last_move_row(game: Connect4, col: int) -> int:
    for r in range(ROWS):
        if game.board[r, col] != 0:
            return r
    return -1


def animate_drop(win: curses.window, game: Connect4, col: int, player: int, delay: float) -> None:
    target_row = find_last_move_row(game, col)
    if target_row < 0:
        return

    y0 = BOARD_TOP
    x0 = BOARD_LEFT
    ch = " X " if player == 1 else " O "
    attr = curses.color_pair(P1_PAIR if player == 1 else P2_PAIR) | curses.A_BOLD
    frame_delay = min(delay * 0.15, 0.06)

    for r in range(target_row + 1):
        cell_x = x0 + 1 + col * CELL_W
        row_y = y0 + 1 + r * 2
        win.addstr(row_y, cell_x, ch, attr | curses.A_REVERSE)
        win.refresh()
        time.sleep(frame_delay)
        if r < target_row:
            win.addstr(row_y, cell_x, " . ", curses.color_pair(EMPTY_PAIR) | curses.A_DIM)


class TrainingVisualizer:
    """Runs parallel training in a background thread and visualizes one game in the TUI.

    Training runs at full speed — never blocks on the display. The TUI polls
    a snapshot queue at its own refresh rate.
    """

    def __init__(self, network: Connect4Net, device: torch.device, num_simulations: int,
                 start_iteration: int = 0, continuous: bool = False) -> None:
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.start_iteration = start_iteration
        self.continuous = continuous

        self.stats: dict = {
            "iteration": 0,
            "game_num": 0,
            "games_total": GAMES_PER_ITERATION,
            "phase": "starting",
            "policy_loss": None,
            "value_loss": None,
            "buffer_size": 0,
            "move_num": 0,
            "current_player": 1,
            "game_status": "",
            "wins": {1: 0, -1: 0, 0: 0},
        }

        self.current_game: Connect4 = Connect4()
        self.current_policy: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_player: int = 1

        # Non-blocking queue of display frames from the training thread.
        # Training pushes snapshots, TUI pops them at its own pace.
        import queue
        self.frame_queue: queue.Queue = queue.Queue(maxsize=500)

        self.quit_flag = threading.Event()
        self.training_done = threading.Event()

    def _on_move(self, game_idx: int, game: Connect4, policy: np.ndarray,
                 action: int, player: int) -> None:
        """Non-blocking callback — pushes a snapshot to the queue."""
        frame = {
            "game": game,
            "policy": policy,
            "action": action,
            "player": player,
            "move_num": int(np.count_nonzero(game.board)),
            "current_player": game.current_player,
            "terminal": game.is_terminal(),
            "winner": game.winner() if game.is_terminal() else None,
        }
        try:
            self.frame_queue.put_nowait(frame)
        except Exception:
            # Queue full — drop frame, training doesn't slow down
            pass

    def training_thread(self) -> None:
        """Run the full AlphaZero training loop with parallel self-play. Never blocks on TUI."""
        optimizer = Adam(self.network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_SIZE)

        iteration = self.start_iteration
        end = float("inf") if self.continuous else self.start_iteration + NUM_ITERATIONS
        while iteration < end and not self.quit_flag.is_set():
            iteration += 1

            self.stats["iteration"] = iteration
            self.stats["phase"] = "self-play"
            self.stats["wins"] = {1: 0, -1: 0, 0: 0}

            parallel = ParallelSelfPlay(
                self.network,
                num_parallel=NUM_PARALLEL,
                num_simulations=self.num_simulations,
                c_puct=C_PUCT,
                temperature_threshold=TEMPERATURE_THRESHOLD,
                device=self.device,
                use_fp16=True,
            )

            # Play games in batches of NUM_PARALLEL
            games_played = 0
            while games_played < GAMES_PER_ITERATION and not self.quit_flag.is_set():
                batch_size = min(NUM_PARALLEL, GAMES_PER_ITERATION - games_played)
                parallel.num_parallel = batch_size

                self.stats["game_num"] = games_played + batch_size

                all_data, finished_games = parallel.play_games(
                    on_move=self._on_move,
                    display_game_idx=0,
                )

                for i, game_data in enumerate(all_data):
                    replay_buffer.extend(game_data)

                    winner = finished_games[i].winner()
                    if winner is None:
                        self.stats["wins"][0] += 1
                    elif winner == 1:
                        self.stats["wins"][1] += 1
                    else:
                        self.stats["wins"][-1] += 1

                games_played += batch_size
                self.stats["buffer_size"] = len(replay_buffer)

            # Training phase
            if len(replay_buffer) >= BATCH_SIZE and not self.quit_flag.is_set():
                self.stats["phase"] = "training"
                self.stats["game_status"] = "Training..."

                p_loss, v_loss = train_network(self.network, optimizer, replay_buffer, self.device)
                self.stats["policy_loss"] = p_loss
                self.stats["value_loss"] = v_loss

            # Checkpoint
            if not self.quit_flag.is_set():
                CHECKPOINT_DIR.mkdir(exist_ok=True)
                ckpt = {
                    "iteration": iteration,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "buffer_size": len(replay_buffer),
                    "num_res_blocks": NUM_RES_BLOCKS,
                    "channels": NUM_CHANNELS,
                }
                torch.save(ckpt, CHECKPOINT_DIR / f"model_iter_{iteration:03d}.pt")
                torch.save(ckpt, CHECKPOINT_DIR / "latest.pt")

        self.stats["phase"] = "done"
        self.stats["game_status"] = "Training complete!"
        self.training_done.set()

    def run_tui(self, stdscr: curses.window, move_delay: float) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)
        init_colors()

        paused = False
        last_col = -1
        last_row = -1
        last_draw_time = 0.0

        while True:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                self.quit_flag.set()
                return
            if key == ord(" "):
                paused = not paused
            if key == ord("+") or key == ord("="):
                move_delay = max(0.05, move_delay - 0.1)
            if key == ord("-"):
                move_delay = min(5.0, move_delay + 0.1)

            now = time.monotonic()

            # Drain the queue — if not paused, pop one frame per refresh interval
            if not paused and (now - last_draw_time) >= move_delay:
                frame = None
                try:
                    frame = self.frame_queue.get_nowait()
                except Exception:
                    pass

                if frame is not None:
                    self.current_game = frame["game"]
                    self.current_policy = frame["policy"]
                    self.last_action = frame["action"]
                    self.last_player = frame["player"]
                    self.stats["move_num"] = frame["move_num"]
                    self.stats["current_player"] = frame["current_player"]

                    if frame["terminal"]:
                        w = frame["winner"]
                        if w is None:
                            self.stats["game_status"] = "DRAW"
                        elif w == 1:
                            self.stats["game_status"] = "X WINS!"
                        else:
                            self.stats["game_status"] = "O WINS!"
                    else:
                        self.stats["game_status"] = ""

                    last_draw_time = now

            # Draw
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            draw_training_header(stdscr, self.stats)

            if self.last_action is not None:
                last_col = self.last_action
                last_row = find_last_move_row(self.current_game, last_col)

            draw_board(stdscr, self.current_game, last_row, last_col)
            if self.current_policy is not None:
                draw_policy_bars(stdscr, self.current_policy, self.current_game)
            draw_footer(stdscr, max_y)

            if paused:
                stdscr.addstr(max_y - 2, BOARD_LEFT, "PAUSED",
                              curses.color_pair(HIGHLIGHT_PAIR) | curses.A_BOLD)

            # Show queue depth so you can see training is ahead of display
            qsize = self.frame_queue.qsize()
            if qsize > 0:
                stdscr.addstr(max_y - 2, BOARD_LEFT + 40, f"Buffered: {qsize} moves",
                              curses.color_pair(DIM_PAIR) | curses.A_DIM)

            stdscr.refresh()

            if self.training_done.is_set() and self.frame_queue.empty():
                stdscr.erase()
                draw_training_header(stdscr, self.stats)
                draw_board(stdscr, self.current_game, last_row, last_col)
                draw_footer(stdscr, max_y)
                stdscr.addstr(max_y - 3, BOARD_LEFT,
                              "Training complete! Press Q to exit.",
                              curses.color_pair(HIGHLIGHT_PAIR) | curses.A_BOLD)
                stdscr.refresh()
                stdscr.nodelay(False)
                while True:
                    key = stdscr.getch()
                    if key in (ord("q"), ord("Q")):
                        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch AlphaZero training live")
    parser.add_argument("--simulations", "-s", type=int, default=NUM_SIMULATIONS,
                        help="MCTS simulations per move")
    parser.add_argument("--delay", "-d", type=float, default=0.5,
                        help="Seconds between moves (adjustable with +/-)")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--continuous", action="store_true",
                        help="Train indefinitely until you press Q")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    network = Connect4Net(num_res_blocks=NUM_RES_BLOCKS, channels=NUM_CHANNELS).to(device)

    # Auto-resume from latest checkpoint, or explicit one
    ckpt_path = args.checkpoint or str(CHECKPOINT_DIR / "latest.pt")
    start_iteration = 0
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        # Check architecture compatibility
        ckpt_blocks = ckpt.get("num_res_blocks", None)
        ckpt_channels = ckpt.get("channels", None)
        if (ckpt_blocks is not None and ckpt_blocks != NUM_RES_BLOCKS) or \
           (ckpt_channels is not None and ckpt_channels != NUM_CHANNELS):
            print(f"Checkpoint architecture mismatch "
                  f"(ckpt: {ckpt_blocks}blocks/{ckpt_channels}ch, "
                  f"current: {NUM_RES_BLOCKS}blocks/{NUM_CHANNELS}ch). Starting fresh.")
        else:
            try:
                network.load_state_dict(ckpt["model_state_dict"])
                start_iteration = ckpt["iteration"]
                print(f"Resuming from iteration {start_iteration}")
            except RuntimeError as e:
                print(f"Checkpoint incompatible, starting fresh: {e}")

    viz = TrainingVisualizer(network, device, args.simulations, start_iteration, args.continuous)

    train_thread = threading.Thread(target=viz.training_thread, daemon=True)
    train_thread.start()

    curses.wrapper(lambda stdscr: viz.run_tui(stdscr, args.delay))

    viz.quit_flag.set()
    train_thread.join(timeout=5)
    print("Done!")


if __name__ == "__main__":
    main()
