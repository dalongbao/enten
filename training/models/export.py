"""
ONNX export utility for trained fish policy.

Exports the policy network for browser inference using ONNX.js.
Only exports the action output (deterministic), not value head.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .policy import FishPolicy


class PolicyForExport(nn.Module):
    """Wrapper that outputs only bounded actions for inference."""

    def __init__(self, policy: FishPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning bounded actions only.

        Args:
            obs: (batch, 56) observation tensor

        Returns:
            action: (batch, 3) bounded actions [speed, direction, urgency]
                - speed in [0, 1]
                - direction in [-1, 1]
                - urgency in [0, 1]
        """
        action_mean, _, _ = self.policy.forward(obs)

        # Apply bounds for deterministic action
        action = torch.stack(
            [
                torch.sigmoid(action_mean[:, 0]),  # speed: [0, 1]
                torch.tanh(action_mean[:, 1]),     # direction: [-1, 1]
                torch.sigmoid(action_mean[:, 2]), # urgency: [0, 1]
            ],
            dim=1,
        )

        return action


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
) -> None:
    """
    Export trained policy to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt file)
        output_path: Path for output ONNX model
        opset_version: ONNX opset version (default 14 for good ONNX.js support)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Create policy and load weights
    policy = FishPolicy()

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
    elif "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        # Assume checkpoint is just the state dict
        policy.load_state_dict(checkpoint)

    policy.eval()

    # Wrap for export
    export_model = PolicyForExport(policy)
    export_model.eval()

    # Create dummy input (batch_size=1, obs_dim=56)
    dummy_input = torch.randn(1, 56)

    # Export to ONNX
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    print(f"Exported model to {output_path}")

    # Verify the exported model
    try:
        import onnx

        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("ONNX model validation passed")

        # Print model info
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

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(args.checkpoint, str(output_path), args.opset)


if __name__ == "__main__":
    main()
