"""Parallel self-play with batched neural network inference.

Runs N games simultaneously. Each MCTS simulation step across all games
collects leaf nodes, batches their evaluations into a single GPU call.
"""

import numpy as np
import torch

from .game import COLS, Connect4
from .mcts import MCTSNode
from .network import Connect4Net


class ParallelSelfPlay:
    """Play multiple games in parallel with batched network inference."""

    def __init__(
        self,
        network: Connect4Net,
        num_parallel: int = 16,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        temperature_threshold: int = 15,
        device: torch.device | None = None,
        use_fp16: bool = True,
    ) -> None:
        self.network = network
        self.num_parallel = num_parallel
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature_threshold = temperature_threshold
        self.device = device or torch.device("cpu")
        self.use_fp16 = use_fp16 and device is not None and device.type in ("cuda", "mps")

    def _batch_evaluate(self, games: list[Connect4]) -> list[tuple[np.ndarray, float]]:
        """Evaluate multiple game states in a single GPU call."""
        if not games:
            return []

        encoded = np.stack([g.encode() for g in games])
        tensor = torch.from_numpy(encoded).to(self.device)

        self.network.eval()
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    p_logits, v = self.network(tensor)
            else:
                p_logits, v = self.network(tensor)

            policies = torch.softmax(p_logits.float(), dim=1).cpu().numpy()
            values = v.float().squeeze(1).cpu().numpy()

        return [(policies[i], float(values[i])) for i in range(len(games))]

    def _run_mcts_batched(self, roots: list[MCTSNode]) -> None:
        """Run MCTS simulations across multiple roots with batched leaf evaluation."""
        # Expand all roots first
        root_games = [r.game for r in roots]
        root_evals = self._batch_evaluate(root_games)
        for root, (policy, _) in zip(roots, root_evals):
            root.expand(policy)

        for _ in range(self.num_simulations):
            # Selection phase: find a leaf in each tree
            leaves: list[MCTSNode] = []
            leaf_indices: list[int] = []  # which root this leaf belongs to
            terminal_leaves: list[tuple[MCTSNode, float]] = []

            for i, root in enumerate(roots):
                node = root
                while node.is_expanded() and not node.game.is_terminal():
                    node = node.select_child(self.c_puct)

                if node.game.is_terminal():
                    winner = node.game.winner()
                    if winner is None:
                        value = 0.0
                    else:
                        value = 1.0 if winner == node.game.current_player else -1.0
                    terminal_leaves.append((node, value))
                else:
                    leaves.append(node)
                    leaf_indices.append(i)

            # Backprop terminals immediately
            for node, value in terminal_leaves:
                node.backpropagate(value)

            # Batch evaluate all non-terminal leaves
            if leaves:
                leaf_games = [leaf.game for leaf in leaves]
                evals = self._batch_evaluate(leaf_games)
                for leaf, (policy, value) in zip(leaves, evals):
                    leaf.expand(policy)
                    leaf.backpropagate(value)

    def play_games(
        self,
        on_move: "callable | None" = None,
        display_game_idx: int = 0,
    ) -> list[list[tuple[np.ndarray, np.ndarray, int]]]:
        """Play num_parallel games simultaneously.

        Args:
            on_move: Optional callback(game_idx, game, policy, action, player)
                     called after each move in the display game.
            display_game_idx: Which game to send to on_move.

        Returns:
            List of training data per game: [(state, policy, outcome), ...]
        """
        # Initialize all games
        games = [Connect4() for _ in range(self.num_parallel)]
        histories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(self.num_parallel)]
        move_counts = [0] * self.num_parallel
        finished = [False] * self.num_parallel

        while not all(finished):
            # Collect active games
            active_indices = [i for i in range(self.num_parallel) if not finished[i]]
            if not active_indices:
                break

            active_games = [games[i] for i in active_indices]

            # Build MCTS roots for all active games
            roots = [MCTSNode(g.copy()) for g in active_games]

            # Run batched MCTS
            self._run_mcts_batched(roots)

            # Extract policies and play moves
            for j, i in enumerate(active_indices):
                game = games[i]
                root = roots[j]
                move_num = move_counts[i]

                # Build visit-count policy
                visits = np.zeros(COLS, dtype=np.float32)
                for child in root.children:
                    visits[child.action] = child.visit_count

                temp = 1.0 if move_num < self.temperature_threshold else 0.0

                if temp == 0:
                    best = int(np.argmax(visits))
                    policy = np.zeros(COLS, dtype=np.float32)
                    policy[best] = 1.0
                else:
                    visits_temp = visits ** (1.0 / temp)
                    total = visits_temp.sum()
                    policy = visits_temp / total if total > 0 else np.ones(COLS, dtype=np.float32) / COLS

                # Record training data
                histories[i].append((game.encode(), policy, game.current_player))

                # Pick action
                if temp > 0:
                    action = int(np.random.choice(COLS, p=policy))
                else:
                    action = int(np.argmax(policy))

                player = game.current_player
                game.play(action)
                move_counts[i] += 1

                # Callback for display game
                if on_move is not None and i == display_game_idx:
                    on_move(i, game.copy(), policy, action, player)

                if game.is_terminal():
                    finished[i] = True
                    if on_move is not None and i == display_game_idx:
                        on_move(i, game.copy(), policy, action, player)

        # Assign outcomes
        all_training_data = []
        for i in range(self.num_parallel):
            game = games[i]
            winner = game.winner()
            training_data = []
            for state, policy, p in histories[i]:
                if winner is None:
                    outcome = 0
                elif winner == p:
                    outcome = 1
                else:
                    outcome = -1
                training_data.append((state, policy, outcome))
            all_training_data.append(training_data)

        return all_training_data, games
