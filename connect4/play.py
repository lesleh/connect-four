"""Play against the trained model or watch it play against a random opponent."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .game import COLS, Connect4
from .mcts import MCTS
from .network import Connect4Net
from .train import NUM_CHANNELS, NUM_RES_BLOCKS


def load_model(checkpoint_path: str, device: torch.device) -> Connect4Net:
    network = Connect4Net(num_res_blocks=NUM_RES_BLOCKS, channels=NUM_CHANNELS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(ckpt["model_state_dict"])
    network.eval()
    print(f"Loaded model from iteration {ckpt['iteration']}")
    return network


def human_vs_ai(network: Connect4Net, device: torch.device, num_simulations: int, human_first: bool) -> None:
    mcts = MCTS(network, num_simulations=num_simulations, device=device)
    game = Connect4()

    human_player = 1 if human_first else -1
    print(f"\nYou are {'X' if human_first else 'O'}. AI is {'O' if human_first else 'X'}.")
    print("Enter column number (0-6) to play.\n")

    while not game.is_terminal():
        print(game)
        print()

        if game.current_player == human_player:
            legal = game.legal_moves()
            while True:
                try:
                    col = int(input("Your move: "))
                    if col in legal:
                        break
                    print(f"Column {col} is not legal. Legal moves: {legal}")
                except (ValueError, EOFError):
                    print(f"Enter a number from {legal}")
        else:
            print("AI is thinking...")
            policy = mcts.search(game, temperature=0)
            col = int(np.argmax(policy))
            print(f"AI plays column {col}")

        game.play(col)

    print(game)
    winner = game.winner()
    if winner is None:
        print("\nDraw!")
    elif winner == human_player:
        print("\nYou win!")
    else:
        print("\nAI wins!")


def ai_vs_random(network: Connect4Net, device: torch.device, num_simulations: int, num_games: int) -> None:
    """Evaluate the AI against a random player."""
    mcts = MCTS(network, num_simulations=num_simulations, device=device)
    wins = draws = losses = 0

    for g in range(num_games):
        game = Connect4()
        ai_player = 1 if g % 2 == 0 else -1  # alternate sides

        while not game.is_terminal():
            if game.current_player == ai_player:
                policy = mcts.search(game, temperature=0)
                action = int(np.argmax(policy))
            else:
                action = np.random.choice(game.legal_moves())
            game.play(action)

        winner = game.winner()
        if winner == ai_player:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

    total = wins + draws + losses
    print(f"\nResults vs random ({total} games):")
    print(f"  Wins:   {wins:4d} ({100*wins/total:.1f}%)")
    print(f"  Draws:  {draws:4d} ({100*draws/total:.1f}%)")
    print(f"  Losses: {losses:4d} ({100*losses/total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect 4 against AlphaZero AI")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/latest.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--simulations", "-s", type=int, default=200,
                        help="MCTS simulations per move")
    parser.add_argument("--mode", "-m", choices=["play", "eval"], default="play",
                        help="'play' for human vs AI, 'eval' for AI vs random")
    parser.add_argument("--games", "-g", type=int, default=50,
                        help="Number of evaluation games (eval mode only)")
    parser.add_argument("--ai-first", action="store_true",
                        help="Let AI go first (play mode only)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run training first: python -m connect4.train")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    network = load_model(args.checkpoint, device)

    if args.mode == "play":
        human_vs_ai(network, device, args.simulations, human_first=not args.ai_first)
    else:
        ai_vs_random(network, device, args.simulations, args.games)


if __name__ == "__main__":
    main()
