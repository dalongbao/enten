"""ONNX export utility for trained fish policy."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .policy import FishPolicy


class PolicyForExport(nn.Module):
    def __init__(self, policy: FishPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action_mean, _, _ = self.policy.forward(obs)
        return torch.sigmoid(action_mean)


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    policy = FishPolicy()

    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"])
    elif "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    elif "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)

    policy.eval()

    export_model = PolicyForExport(policy)
    export_model.eval()

    dummy_input = torch.randn(3, 60)  # Fixed batch size of 3 fish

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
    )

    print(f"Exported model to {output_path}")

    try:
        import onnx

        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("ONNX model validation passed")
        print(f"  Inputs: {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")

    except ImportError:
        print("Note: Install 'onnx' package to verify exported model")


def main():
    parser = argparse.ArgumentParser(description="Export fish policy to ONNX")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="inference/model.onnx",
        help="Output ONNX file path (default: inference/model.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(args.checkpoint, str(output_path), args.opset)


if __name__ == "__main__":
    main()
