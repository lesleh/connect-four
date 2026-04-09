"""Monte Carlo Tree Search with PUCT selection (AlphaZero-style)."""

import math

import numpy as np
import torch

from .game import COLS, Connect4
from .network import Connect4Net


class MCTSNode:
    __slots__ = ("game", "parent", "action", "prior", "children",
                 "visit_count", "value_sum")

    def __init__(
        self,
        game: Connect4,
        parent: "MCTSNode | None" = None,
        action: int | None = None,
        prior: float = 0.0,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: list[MCTSNode] = []
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def select_child(self, c_puct: float = 1.5) -> "MCTSNode":
        """Select child with highest PUCT score."""
        total_visits = sum(c.visit_count for c in self.children)
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = -float("inf")
        best_child = self.children[0]
        for child in self.children:
            q = -child.q_value  # negate: child stores value for child's player, but parent wants opponent's loss
            u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, policy: np.ndarray) -> None:
        """Create child nodes for all legal moves, weighted by policy."""
        legal = self.game.legal_moves()
        # Mask illegal moves and renormalize
        masked = np.zeros(COLS, dtype=np.float32)
        for col in legal:
            masked[col] = policy[col]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            # Uniform fallback
            for col in legal:
                masked[col] = 1.0 / len(legal)

        for col in legal:
            child_game = self.game.copy()
            child_game.play(col)
            child = MCTSNode(child_game, parent=self, action=col, prior=masked[col])
            self.children.append(child)

    def backpropagate(self, value: float) -> None:
        """Propagate value up the tree, flipping sign at each level."""
        node: MCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # opponent's perspective
            node = node.parent


class MCTS:
    """Run MCTS simulations from a root game state."""

    def __init__(
        self,
        network: Connect4Net,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        device: torch.device | None = None,
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device or torch.device("cpu")

    def search(self, game: Connect4, temperature: float = 1.0) -> np.ndarray:
        """Run MCTS and return visit-count policy over columns."""
        root = MCTSNode(game.copy())

        # Expand root
        policy, _ = self._evaluate(root.game)
        root.expand(policy)

        for _ in range(self.num_simulations):
            node = root

            # Selection — descend until we hit an unexpanded or terminal node
            while node.is_expanded() and not node.game.is_terminal():
                node = node.select_child(self.c_puct)

            # If terminal, backpropagate the true outcome
            if node.game.is_terminal():
                winner = node.game.winner()
                if winner is None:
                    value = 0.0  # draw
                else:
                    # Value from the perspective of node's *parent's* player
                    # node.game.current_player has already flipped after the last move,
                    # so if winner == node.game.current_player, the current player won
                    # — but actually the last move was made by the opponent of current_player
                    value = 1.0 if winner == node.game.current_player else -1.0
                node.backpropagate(value)
                continue

            # Expansion + evaluation
            policy, value = self._evaluate(node.game)
            node.expand(policy)
            node.backpropagate(value)

        # Build visit-count distribution
        visits = np.zeros(COLS, dtype=np.float32)
        for child in root.children:
            visits[child.action] = child.visit_count

        if temperature == 0:
            # Greedy
            best = np.argmax(visits)
            probs = np.zeros(COLS, dtype=np.float32)
            probs[best] = 1.0
            return probs

        # Apply temperature
        visits_temp = visits ** (1.0 / temperature)
        total = visits_temp.sum()
        if total > 0:
            return visits_temp / total
        # Fallback to uniform over legal moves
        legal = game.legal_moves()
        probs = np.zeros(COLS, dtype=np.float32)
        for c in legal:
            probs[c] = 1.0 / len(legal)
        return probs

    def _evaluate(self, game: Connect4) -> tuple[np.ndarray, float]:
        """Get policy and value from the neural network."""
        encoded = torch.from_numpy(game.encode()).to(self.device)
        policy, value = self.network.predict(encoded)
        return policy.cpu().numpy(), value
