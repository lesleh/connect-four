"""Export trained Connect4 model to ONNX for browser inference."""

import argparse
from pathlib import Path

import onnx
import torch

from connect4.game import GameConfig
from connect4.network import Connect4Net
from connect4.train import NUM_RES_BLOCKS, NUM_CHANNELS


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--win", type=int, default=None)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        rows = args.rows or 6
        cols = args.cols or 7
        win = args.win or 4
        ckpt_path = Path(f"checkpoints/{win}_{rows}x{cols}/latest.pt")

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    rows = ckpt.get("rows", 6)
    cols = ckpt.get("cols", 7)

    net = Connect4Net(
        rows=rows,
        cols=cols,
        num_res_blocks=ckpt.get("num_res_blocks", NUM_RES_BLOCKS),
        channels=ckpt.get("channels", NUM_CHANNELS),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    win_length = ckpt.get("win_length", 4)
    dummy = torch.randn(1, 2, rows, cols)

    output = Path(args.output) if args.output else Path(f"web/public/models/{win_length}_{rows}x{cols}.onnx")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        str(output),
        input_names=["board"],
        output_names=["policy_logits", "value"],
    )

    # Merge external data into single file (needed for browser)
    data_file = Path(str(output) + ".data")
    if data_file.exists():
        model = onnx.load(str(output), load_external_data=True)
        onnx.save(model, str(output))
        data_file.unlink()

    print(f"Exported to {output} ({output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
