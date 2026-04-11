import * as ort from "onnxruntime-web";
import type { GameConfig } from "../game/connect4";

let session: ort.InferenceSession | null = null;
let currentModel: string | null = null;

function modelPath(config: GameConfig): string {
  return `/models/${config.winLength}_${config.rows}x${config.cols}.onnx`;
}

export async function loadModel(config: GameConfig): Promise<void> {
  const path = modelPath(config);
  if (currentModel === path && session) return;

  session = await ort.InferenceSession.create(path, {
    executionProviders: ["webgpu", "wasm"],
  });
  currentModel = path;
}

export async function predict(
  encoded: Float32Array,
  config: GameConfig
): Promise<{ policy: Float32Array; value: number }> {
  if (!session) throw new Error("Model not loaded");

  const input = new ort.Tensor("float32", encoded, [1, 2, config.rows, config.cols]);
  const results = await session.run({ board: input });

  const logits = results.policy_logits.data as Float32Array;
  const value = (results.value.data as Float32Array)[0];

  // Softmax
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - maxLogit);
    sum += exps[i];
  }
  const policy = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    policy[i] = exps[i] / sum;
  }

  return { policy, value };
}

export function isLoaded(): boolean {
  return session !== null;
}
