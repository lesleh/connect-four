"""GPU inference server + remote MCTS for multiprocessing self-play.

Workers do tree traversal on CPU, send leaf evaluations to a central
GPU server in batches. Uses virtual loss to collect multiple leaves
per round-trip, reducing queue overhead.
"""

import math
import threading
import time
from multiprocessing import Queue, Event

import numpy as np
import torch
import torch.nn.functional as F

from .game import Connect4, GameConfig
from .mcts import MCTSNode
from .network import Connect4Net


class InferenceServer:
    """Runs in the main process. Collects eval requests from workers, batches on GPU."""

    def __init__(
        self,
        network: Connect4Net,
        device: torch.device,
        num_workers: int,
        batch_wait: float = 0.002,
        max_batch: int = 128,
        use_fp16: bool = True,
    ) -> None:
        self.network = network
        self.device = device
        self.request_queue: Queue = Queue()
        self.response_queues: list[Queue] = [Queue() for _ in range(num_workers)]
        self.stop_event = Event()
        self.batch_wait = batch_wait
        self.max_batch = max_batch
        self.use_fp16 = use_fp16 and device.type in ("cuda", "mps")
        self._thread: threading.Thread | None = None
        self.batches_processed = 0
        self.total_inferences = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        self.network.eval()
        while not self.stop_event.is_set():
            batch = self._collect_batch()
            if batch:
                self._evaluate_and_respond(batch)

    def _collect_batch(self) -> list:
        batch = []
        try:
            item = self.request_queue.get(timeout=0.05)
            batch.append(item)
        except Exception:
            return batch

        # Collect more requests briefly
        deadline = time.monotonic() + self.batch_wait
        while len(batch) < self.max_batch and time.monotonic() < deadline:
            try:
                item = self.request_queue.get_nowait()
                batch.append(item)
            except Exception:
                # Brief sleep to let more requests arrive
                time.sleep(0.0002)
                try:
                    item = self.request_queue.get_nowait()
                    batch.append(item)
                except Exception:
                    break
        return batch

    def _evaluate_and_respond(self, batch: list) -> None:
        worker_ids = [b[0] for b in batch]
        states = np.stack([b[1] for b in batch])

        with torch.no_grad():
            tensor = torch.from_numpy(states).to(self.device)
            if self.use_fp16:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    p_logits, v = self.network(tensor)
                policies = F.softmax(p_logits.float(), dim=1).cpu().numpy()
                values = v.float().squeeze(1).cpu().numpy()
            else:
                p_logits, v = self.network(tensor)
                policies = F.softmax(p_logits, dim=1).cpu().numpy()
                values = v.squeeze(1).cpu().numpy()

        for i, wid in enumerate(worker_ids):
            self.response_queues[wid].put((policies[i], float(values[i])))

        self.batches_processed += 1
        self.total_inferences += len(batch)


# -- Remote MCTS (runs in worker processes) -------------------

VIRTUAL_LOSS_BATCH = 8  # leaves to collect per round-trip


def _remote_evaluate(
    game: Connect4,
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
) -> tuple[np.ndarray, float]:
    """Send one eval request to the server and wait for response."""
    request_queue.put((worker_id, game.encode()))
    policy, value = response_queue.get()
    return policy, value


def _remote_evaluate_batch(
    games: list[Connect4],
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
) -> list[tuple[np.ndarray, float]]:
    """Send multiple eval requests and collect all responses."""
    for game in games:
        request_queue.put((worker_id, game.encode()))
    results = []
    for _ in games:
        policy, value = response_queue.get()
        results.append((policy, value))
    return results


def remote_mcts_search(
    game: Connect4,
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    temperature: float = 1.0,
) -> np.ndarray:
    """Run MCTS using remote GPU inference with virtual loss batching."""
    cols = game.config.cols
    root = MCTSNode(game.copy())

    # Expand root
    policy, _ = _remote_evaluate(root.game, worker_id, request_queue, response_queue)
    root.expand(policy)

    sims_done = 0
    while sims_done < num_simulations:
        # Collect multiple leaves using virtual loss
        leaves = []
        terminal_backprops = []

        batch_size = min(VIRTUAL_LOSS_BATCH, num_simulations - sims_done)
        for _ in range(batch_size):
            node = root
            while node.is_expanded() and not node.game.is_terminal():
                node = node.select_child(c_puct)

            if node.game.is_terminal():
                winner = node.game.winner()
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if winner == node.game.current_player else -1.0
                terminal_backprops.append((node, value))
                sims_done += 1
                continue

            # Apply virtual loss to steer future selections away
            node.visit_count += 1
            node.value_sum -= 1.0
            leaves.append(node)
            sims_done += 1

        # Backprop terminals immediately
        for node, value in terminal_backprops:
            node.backpropagate(value)

        if not leaves:
            continue

        # Batch evaluate all leaves via GPU server
        leaf_games = [leaf.game for leaf in leaves]
        results = _remote_evaluate_batch(leaf_games, worker_id, request_queue, response_queue)

        for leaf, (policy, value) in zip(leaves, results):
            # Remove virtual loss
            leaf.visit_count -= 1
            leaf.value_sum += 1.0
            # Real expand + backprop
            leaf.expand(policy)
            leaf.backpropagate(value)

    # Build visit-count policy
    visits = np.zeros(cols, dtype=np.float32)
    for child in root.children:
        visits[child.action] = child.visit_count

    if temperature == 0:
        best = int(np.argmax(visits))
        probs = np.zeros(cols, dtype=np.float32)
        probs[best] = 1.0
        return probs

    visits_temp = visits ** (1.0 / temperature)
    total = visits_temp.sum()
    if total > 0:
        return visits_temp / total
    legal = game.legal_moves()
    probs = np.zeros(cols, dtype=np.float32)
    for c in legal:
        probs[c] = 1.0 / len(legal)
    return probs


# -- Worker entry point ----------------------------------------

def worker_play_games(
    worker_id: int,
    num_games: int,
    request_queue: Queue,
    response_queue: Queue,
    result_queue: Queue,
    num_simulations: int,
    c_puct: float,
    temperature_threshold: int,
    rows: int = 6,
    cols: int = 7,
    win_length: int = 4,
) -> None:
    """Worker process: plays games using remote GPU inference."""
    config = GameConfig(rows=rows, cols=cols, win_length=win_length)
    for _ in range(num_games):
        game = Connect4(config)
        history = []
        move_num = 0

        while not game.is_terminal():
            temp = 1.0 if move_num < temperature_threshold else 0.0

            policy = remote_mcts_search(
                game, worker_id, request_queue, response_queue,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temperature=temp,
            )

            history.append((game.encode(), policy, game.current_player))

            if temp > 0:
                action = int(np.random.choice(cols, p=policy))
            else:
                action = int(np.argmax(policy))

            game.play(action)
            move_num += 1

        winner = game.winner()
        training_data = []
        for state, policy, player in history:
            if winner is None:
                outcome = 0
            elif winner == player:
                outcome = 1
            else:
                outcome = -1
            training_data.append((state, policy, outcome))

        result_queue.put((training_data, winner))
