"""AlphaZero self-play training loop with multiprocessing + GPU inference server."""

import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from .game import COLS, Connect4
from .mcts import MCTS
from .minimax import minimax_move
from .network import Connect4Net
from .inference_server import InferenceServer, worker_play_games

EVAL_GAMES = 20
MINIMAX_DEPTH = 5

# ── Hyperparameters ──────────────────────────────────────────────

NUM_ITERATIONS = 50
GAMES_PER_ITERATION = 200
NUM_SIMULATIONS = 200
REPLAY_BUFFER_SIZE = 150_000
BATCH_SIZE = 512
EPOCHS_PER_ITERATION = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE_THRESHOLD = 15
C_PUCT = 1.5
NUM_RES_BLOCKS = 5
NUM_CHANNELS = 128
NUM_WORKERS = max(1, os.cpu_count() - 2)
CHECKPOINT_DIR = Path("checkpoints")


# ── Training ─────────────────────────────────────────────────────

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


def evaluate_vs_minimax(network: Connect4Net, device: torch.device, num_games: int = EVAL_GAMES) -> tuple[int, int, int]:
    """Evaluate AI (200 sims) vs minimax (depth 5). Returns (wins, draws, losses)."""
    mcts = MCTS(network, num_simulations=200, c_puct=C_PUCT, device=device)
    wins = draws = losses = 0
    for g in range(num_games):
        game = Connect4()
        ai_player = 1 if g % 2 == 0 else -1
        while not game.is_terminal():
            if game.current_player == ai_player:
                policy = mcts.search(game, temperature=0)
                action = int(np.argmax(policy))
            else:
                action = minimax_move(game, depth=MINIMAX_DEPTH)
            game.play(action)
        winner = game.winner()
        if winner == ai_player:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def save_checkpoint(network, optimizer, iteration, buffer_size, path):
    torch.save({
        "iteration": iteration,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "buffer_size": buffer_size,
        "num_res_blocks": NUM_RES_BLOCKS,
        "channels": NUM_CHANNELS,
    }, path)


def main() -> None:
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device} (training + inference server)")
    print(f"Self-play workers: {NUM_WORKERS} (CPU tree search → GPU eval)")

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    network = Connect4Net(num_res_blocks=NUM_RES_BLOCKS, channels=NUM_CHANNELS).to(device)
    optimizer = Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    replay_buffer: deque = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Auto-resume
    start_iteration = 0
    latest = CHECKPOINT_DIR / "latest.pt"
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

    for iteration in range(start_iteration + 1, start_iteration + NUM_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")

        # ── Start inference server ───────────────────────────────
        server = InferenceServer(
            network, device, NUM_WORKERS,
            batch_wait=0.003, max_batch=NUM_WORKERS * 8,
            use_fp16=True,
        )
        server.start()

        # ── Self-play (multiprocessing + GPU server) ─────────────
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

        # ── Training (GPU) ───────────────────────────────────────
        if len(replay_buffer) >= BATCH_SIZE:
            p_loss, v_loss = train_network(network, optimizer, replay_buffer, device)
            print(f"Policy loss: {p_loss:.4f}  Value loss: {v_loss:.4f}")

        # ── Evaluate vs random ────────────────────────────────────
        w, d, l = evaluate_vs_minimax(network, device)
        winrate = (w + 0.5 * d) / (w + d + l) * 100
        print(f"Eval vs minimax d{MINIMAX_DEPTH} ({w+d+l}g): {w}W {d}D {l}L  ({winrate:.0f}%)")

        # ── Checkpoint ───────────────────────────────────────────
        ckpt_path = CHECKPOINT_DIR / f"model_iter_{iteration:03d}.pt"
        save_checkpoint(network, optimizer, iteration, len(replay_buffer), ckpt_path)
        save_checkpoint(network, optimizer, iteration, len(replay_buffer), CHECKPOINT_DIR / "latest.pt")
        print(f"Saved checkpoint: {ckpt_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
