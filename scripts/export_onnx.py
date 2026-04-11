"""Export trained Connect4 model to ONNX for browser inference."""

from pathlib import Path

import torch

from connect4.network import Connect4Net
from connect4.train import NUM_RES_BLOCKS, NUM_CHANNELS

CHECKPOINT = Path("checkpoints/latest.pt")
OUTPUT = Path("web/public/model.onnx")


def main():
    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)

    net = Connect4Net(
        num_res_blocks=ckpt.get("num_res_blocks", NUM_RES_BLOCKS),
        channels=ckpt.get("channels", NUM_CHANNELS),
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    dummy = torch.randn(1, 2, 6, 7)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        str(OUTPUT),
        input_names=["board"],
        output_names=["policy_logits", "value"],
    )
    print(f"Exported to {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
