"""Run eval against minimax at any depth without training."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .game import Connect4
from .mcts import MCTS
from .minimax import minimax_move
from .network import Connect4Net
from .train import NUM_RES_BLOCKS, NUM_CHANNELS, C_PUCT


def evaluate(network: Connect4Net, device: torch.device, depth: int, num_games: int, sims: int) -> None:
    mcts = MCTS(network, num_simulations=sims, c_puct=C_PUCT, device=device)
    p1 = {"wins": 0, "draws": 0, "losses": 0}
    p2 = {"wins": 0, "draws": 0, "losses": 0}

    for g in range(num_games):
        game = Connect4()
        ai_player = 1 if g % 2 == 0 else -1
        bucket = p1 if ai_player == 1 else p2
        side = "P1" if ai_player == 1 else "P2"

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
            result = "WIN"
        elif winner is None:
            bucket["draws"] += 1
            result = "DRAW"
        else:
            bucket["losses"] += 1
            result = "LOSS"

        total = g + 1
        print(f"Game {total}/{num_games} ({side}): {result}")

    total_w = p1["wins"] + p2["wins"]
    total_d = p1["draws"] + p2["draws"]
    total_l = p1["losses"] + p2["losses"]
    total = total_w + total_d + total_l
    winrate = (total_w + 0.5 * total_d) / total * 100

    print(f"\nResults vs minimax d{depth} ({total}g, {sims} MCTS sims):")
    print(f"  P1: {p1['wins']}W {p1['draws']}D {p1['losses']}L")
    print(f"  P2: {p2['wins']}W {p2['draws']}D {p2['losses']}L")
    print(f"  Total: {total_w}W {total_d}D {total_l}L ({winrate:.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI vs minimax")
    parser.add_argument("-d", "--depth", type=int, default=5, help="Minimax depth (default: 5)")
    parser.add_argument("-n", "--games", type=int, default=20, help="Number of games (default: 20)")
    parser.add_argument("-s", "--sims", type=int, default=200, help="MCTS simulations (default: 200)")
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/latest.pt", help="Checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    network = Connect4Net(
        num_res_blocks=ckpt.get("num_res_blocks", NUM_RES_BLOCKS),
        channels=ckpt.get("channels", NUM_CHANNELS),
    ).to(device)
    network.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {ckpt_path} (iteration {ckpt.get('iteration', '?')})")

    evaluate(network, device, args.depth, args.games, args.sims)


if __name__ == "__main__":
    main()
