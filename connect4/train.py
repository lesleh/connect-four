"""AlphaZero self-play training loop with multiprocessing + GPU inference server."""

import argparse
import json
import os
import pickle
import signal
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from .game import Connect4, GameConfig
from .mcts import MCTS
from .minimax import minimax_move
from .network import Connect4Net
from .inference_server import InferenceServer, worker_play_games

EVAL_GAMES = 20
EVAL_DEPTHS = [6, 7]

# -- Hyperparameters --

NUM_ITERATIONS = 150
GAMES_PER_ITERATION = 200
MINIMAX_GAMES_PER_ITERATION = 40  # 20% minimax opponent games
MINIMAX_TRAIN_DEPTHS = [3, 4, 5]
NUM_SIMULATIONS = 400
REPLAY_BUFFER_SIZE = 150_000
BATCH_SIZE = 256
EPOCHS_PER_ITERATION = 30
LEARNING_RATE = 1e-3
LR_DECAY = 0.99  # multiply LR by this each iteration
WEIGHT_DECAY = 1e-4
TEMPERATURE_THRESHOLD = 15
C_PUCT = 1.5
NUM_RES_BLOCKS = 5
NUM_CHANNELS = 128
NUM_WORKERS = max(1, os.cpu_count() - 2)


# -- Training --

def train_network(
    network: Connect4Net,
    optimizer: torch.optim.Optimizer,
    replay_buffer: deque,
    device: torch.device,
) -> tuple[float, float]:
    network.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for _ in range(EPOCHS_PER_ITERATION):
        indices = np.random.choice(len(replay_buffer), size=min(BATCH_SIZE, len(replay_buffer)), replace=False)
        batch = [replay_buffer[i] for i in indices]

        states = torch.from_numpy(np.array([b[0] for b in batch])).to(device)
        target_policies = torch.from_numpy(np.array([b[1] for b in batch])).to(device)
        target_values = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(device)

        policy_logits, values = network(states)

        policy_loss = -torch.mean(torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = F.mse_loss(values, target_values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return total_policy_loss / num_batches, total_value_loss / num_batches


def evaluate_vs_minimax(network: Connect4Net, device: torch.device, depth: int, config: GameConfig, num_games: int = EVAL_GAMES) -> dict:
    """Evaluate AI (200 sims) vs minimax at given depth. Returns results split by P1/P2."""
    mcts = MCTS(network, num_simulations=200, c_puct=C_PUCT, device=device)
    p1 = {"wins": 0, "draws": 0, "losses": 0}
    p2 = {"wins": 0, "draws": 0, "losses": 0}
    for g in range(num_games):
        game = Connect4(config)
        ai_player = 1 if g % 2 == 0 else -1
        bucket = p1 if ai_player == 1 else p2
        while not game.is_terminal():
            if game.current_player == ai_player:
                policy = mcts.search(game, temperature=0)
                action = int(np.argmax(policy))
            else:
                action = minimax_move(game, depth=depth)
            game.play(action)
        winner = game.winner()
        if winner == ai_player:
            bucket["wins"] += 1
        elif winner is None:
            bucket["draws"] += 1
        else:
            bucket["losses"] += 1
    return {"p1": p1, "p2": p2}


def play_vs_minimax(network: Connect4Net, device: torch.device, num_games: int, depths: list[int], config: GameConfig) -> list:
    """Play games vs minimax at random depths, return training data for the AI side."""
    mcts = MCTS(network, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT, device=device)
    training_data = []
    for g in range(num_games):
        game = Connect4(config)
        depth = depths[g % len(depths)]
        ai_player = 1 if g % 2 == 0 else -1
        history = []
        move_num = 0

        while not game.is_terminal():
            if game.current_player == ai_player:
                temp = 1.0 if move_num < TEMPERATURE_THRESHOLD else 0.0
                policy = mcts.search(game, temperature=temp)
                history.append((game.encode(), policy, game.current_player))
                if temp > 0:
                    action = int(np.random.choice(config.cols, p=policy))
                else:
                    action = int(np.argmax(policy))
            else:
                action = minimax_move(game, depth=depth)
            game.play(action)
            move_num += 1

        winner = game.winner()
        for state, policy, player in history:
            if winner is None:
                outcome = 0
            elif winner == player:
                outcome = 1
            else:
                outcome = -1
            training_data.append((state, policy, outcome))

    return training_data


def save_checkpoint(network, optimizer, iteration, buffer_size, config, path):
    torch.save({
        "iteration": iteration,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "buffer_size": buffer_size,
        "num_res_blocks": NUM_RES_BLOCKS,
        "channels": NUM_CHANNELS,
        "rows": config.rows,
        "cols": config.cols,
        "win_length": config.win_length,
    }, path)


def save_buffer(replay_buffer: deque, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(list(replay_buffer), f)


def load_buffer(path: Path, maxlen: int) -> deque:
    with open(path, "rb") as f:
        data = pickle.load(f)
    buf = deque(data, maxlen=maxlen)
    print(f"Loaded replay buffer: {len(buf)} examples")
    return buf


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero training")
    parser.add_argument("--rows", type=int, default=6, help="Board rows (default: 6)")
    parser.add_argument("--cols", type=int, default=7, help="Board columns (default: 7)")
    parser.add_argument("--win", type=int, default=4, help="Win length (default: 4)")
    args = parser.parse_args()

    config = GameConfig(rows=args.rows, cols=args.cols, win_length=args.win)
    checkpoint_dir = Path(f"checkpoints/{config.win_length}_{config.rows}x{config.cols}")

    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Game: Connect {config.win_length} on {config.rows}x{config.cols} board")
    print(f"Using device: {device} (training + inference server)")
    print(f"Self-play workers: {NUM_WORKERS} (CPU tree search -> GPU eval)")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    network = Connect4Net(rows=config.rows, cols=config.cols, num_res_blocks=NUM_RES_BLOCKS, channels=NUM_CHANNELS).to(device)
    optimizer = Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_SIZE)
    buffer_path = checkpoint_dir / "replay_buffer.pkl"
    if buffer_path.exists():
        replay_buffer = load_buffer(buffer_path, REPLAY_BUFFER_SIZE)

    # Auto-resume
    start_iteration = 0
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        ckpt_blocks = ckpt.get("num_res_blocks", None)
        ckpt_channels = ckpt.get("channels", None)
        if (ckpt_blocks is not None and ckpt_blocks != NUM_RES_BLOCKS) or \
           (ckpt_channels is not None and ckpt_channels != NUM_CHANNELS):
            print("Checkpoint architecture mismatch, starting fresh.")
        else:
            try:
                network.load_state_dict(ckpt["model_state_dict"])
                start_iteration = ckpt["iteration"]
                print(f"Resuming from iteration {start_iteration}")
            except RuntimeError:
                print("Checkpoint incompatible, starting fresh.")

    # Save buffer on ctrl+c
    def handle_interrupt(signum, frame):
        print("\n\nInterrupted! Saving replay buffer...")
        save_buffer(replay_buffer, buffer_path)
        print(f"Buffer saved ({len(replay_buffer)} examples). Exiting.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    for iteration in range(start_iteration + 1, start_iteration + NUM_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")

        # -- Start inference server --
        server = InferenceServer(
            network, device, NUM_WORKERS,
            batch_wait=0.003, max_batch=NUM_WORKERS * 8,
            use_fp16=True,
        )
        server.start()

        # -- Self-play (multiprocessing + GPU server) --
        games_per_worker = [GAMES_PER_ITERATION // NUM_WORKERS] * NUM_WORKERS
        for i in range(GAMES_PER_ITERATION % NUM_WORKERS):
            games_per_worker[i] += 1

        result_queue = mp.Queue()
        workers = []
        for w_id in range(NUM_WORKERS):
            if games_per_worker[w_id] == 0:
                continue
            p = mp.Process(
                target=worker_play_games,
                args=(
                    w_id,
                    games_per_worker[w_id],
                    server.request_queue,
                    server.response_queues[w_id],
                    result_queue,
                    NUM_SIMULATIONS,
                    C_PUCT,
                    TEMPERATURE_THRESHOLD,
                    config.rows,
                    config.cols,
                    config.win_length,
                ),
            )
            p.start()
            workers.append(p)

        # Collect results
        total_games = GAMES_PER_ITERATION
        new_examples = 0
        wins = {1: 0, -1: 0, 0: 0}

        with tqdm(total=total_games, desc="Self-play") as pbar:
            collected = 0
            while collected < total_games:
                training_data, winner = result_queue.get()
                replay_buffer.extend(training_data)
                new_examples += len(training_data)
                if winner is None:
                    wins[0] += 1
                elif winner == 1:
                    wins[1] += 1
                else:
                    wins[-1] += 1
                collected += 1
                pbar.update(1)

        for p in workers:
            p.join()

        # Stop inference server
        server.stop()
        print(f"Generated {new_examples} examples from {total_games} games "
              f"(X:{wins[1]} O:{wins[-1]} D:{wins[0]})  "
              f"Buffer: {len(replay_buffer)}")
        print(f"GPU server: {server.batches_processed} batches, "
              f"{server.total_inferences} inferences "
              f"(avg batch size: {server.total_inferences / max(1, server.batches_processed):.1f})")

        # -- Play vs minimax (single-process, on GPU) --
        mm_data = play_vs_minimax(network, device, MINIMAX_GAMES_PER_ITERATION, MINIMAX_TRAIN_DEPTHS, config)
        replay_buffer.extend(mm_data)
        print(f"Minimax training: {len(mm_data)} examples from {MINIMAX_GAMES_PER_ITERATION} games")

        # -- Training (GPU) --
        if len(replay_buffer) >= BATCH_SIZE:
            p_loss, v_loss = train_network(network, optimizer, replay_buffer, device)
            scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Policy loss: {p_loss:.4f}  Value loss: {v_loss:.4f}  LR: {lr_now:.6f}")

        # -- Evaluate vs minimax at multiple depths --
        eval_results = {}
        for depth in EVAL_DEPTHS:
            result = evaluate_vs_minimax(network, device, depth, config)
            p1, p2 = result["p1"], result["p2"]
            total_w = p1["wins"] + p2["wins"]
            total_d = p1["draws"] + p2["draws"]
            total_l = p1["losses"] + p2["losses"]
            total = total_w + total_d + total_l
            winrate = (total_w + 0.5 * total_d) / total * 100
            eval_results[depth] = {**result, "winrate": winrate}
            print(f"Eval d{depth}: P1 {p1['wins']}W {p1['draws']}D {p1['losses']}L | "
                  f"P2 {p2['wins']}W {p2['draws']}D {p2['losses']}L | "
                  f"Total ({winrate:.0f}%)")

        # -- Log iteration stats --
        log_entry = {
            "iteration": iteration,
            "policy_loss": round(p_loss, 4) if len(replay_buffer) >= BATCH_SIZE else None,
            "value_loss": round(v_loss, 4) if len(replay_buffer) >= BATCH_SIZE else None,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "buffer_size": len(replay_buffer),
            "eval": {str(d): r for d, r in eval_results.items()},
        }
        log_path = checkpoint_dir / "training_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # -- Checkpoint --
        ckpt_path = checkpoint_dir / f"model_iter_{iteration:03d}.pt"
        save_checkpoint(network, optimizer, iteration, len(replay_buffer), config, ckpt_path)
        save_checkpoint(network, optimizer, iteration, len(replay_buffer), config, checkpoint_dir / "latest.pt")
        save_buffer(replay_buffer, buffer_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
