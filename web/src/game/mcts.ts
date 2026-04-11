import { Connect4 } from "./connect4";
import { predict } from "../ai/model";

class MCTSNode {
  game: Connect4;
  parent: MCTSNode | null;
  action: number | null;
  prior: number;
  children: MCTSNode[];
  visitCount: number;
  valueSum: number;

  constructor(
    game: Connect4,
    parent: MCTSNode | null = null,
    action: number | null = null,
    prior: number = 0
  ) {
    this.game = game;
    this.parent = parent;
    this.action = action;
    this.prior = prior;
    this.children = [];
    this.visitCount = 0;
    this.valueSum = 0;
  }

  get qValue(): number {
    if (this.visitCount === 0) return 0;
    return this.valueSum / this.visitCount;
  }

  isExpanded(): boolean {
    return this.children.length > 0;
  }

  selectChild(cPuct: number): MCTSNode {
    let totalVisits = 0;
    for (const c of this.children) totalVisits += c.visitCount;
    const sqrtTotal = Math.sqrt(totalVisits + 1);

    let bestScore = -Infinity;
    let bestChild = this.children[0];
    for (const child of this.children) {
      const q = -child.qValue;
      const u = cPuct * child.prior * sqrtTotal / (1 + child.visitCount);
      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        bestChild = child;
      }
    }
    return bestChild;
  }

  expand(policy: Float32Array): void {
    const cols = this.game.config.cols;
    const legal = this.game.legalMoves();
    const masked = new Float32Array(cols);
    let total = 0;
    for (const col of legal) {
      masked[col] = policy[col];
      total += policy[col];
    }
    if (total > 0) {
      for (const col of legal) masked[col] /= total;
    } else {
      for (const col of legal) masked[col] = 1 / legal.length;
    }

    for (const col of legal) {
      const childGame = this.game.copy();
      childGame.play(col);
      this.children.push(new MCTSNode(childGame, this, col, masked[col]));
    }
  }

  backpropagate(value: number): void {
    let node: MCTSNode | null = this;
    while (node !== null) {
      node.visitCount++;
      node.valueSum += value;
      value = -value;
      node = node.parent;
    }
  }
}

export async function mctsSearch(
  game: Connect4,
  numSimulations: number = 80,
  cPuct: number = 1.5
): Promise<{ policy: Float32Array; visits: Float32Array }> {
  const cols = game.config.cols;
  const config = game.config;
  const root = new MCTSNode(game.copy());

  // Expand root
  const { policy, value: _ } = await predict(root.game.encode(), config);
  root.expand(policy);

  for (let i = 0; i < numSimulations; i++) {
    let node = root;

    // Selection
    while (node.isExpanded() && !node.game.isTerminal()) {
      node = node.selectChild(cPuct);
    }

    // Terminal
    if (node.game.isTerminal()) {
      const winner = node.game.winner();
      let value: number;
      if (winner === null) {
        value = 0;
      } else {
        value = winner === node.game.currentPlayer ? 1 : -1;
      }
      node.backpropagate(value);
      continue;
    }

    // Expand + evaluate
    const result = await predict(node.game.encode(), config);
    node.expand(result.policy);
    node.backpropagate(result.value);
  }

  // Build visit distribution
  const visits = new Float32Array(cols);
  for (const child of root.children) {
    if (child.action !== null) {
      visits[child.action] = child.visitCount;
    }
  }

  // Greedy (temperature 0)
  let bestCol = 0;
  let bestVisits = 0;
  for (let c = 0; c < cols; c++) {
    if (visits[c] > bestVisits) {
      bestVisits = visits[c];
      bestCol = c;
    }
  }
  const probs = new Float32Array(cols);
  probs[bestCol] = 1;

  return { policy: probs, visits };
}
