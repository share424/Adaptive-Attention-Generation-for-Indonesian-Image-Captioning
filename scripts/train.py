"""
This script is used to train the Adaptive Attention Image Captioning model.

Usage:
    python train.py <config> [--device <device>] [--ckpt <checkpoint>]

Arguments:
    <config>                Path to the configuration file.
    --device <device>       Device to use. Default is 'cuda'.
    --ckpt <checkpoint>     Path to the checkpoint file. Optional.

Example:
    python train.py config.yaml --device cuda --ckpt checkpoint.pth

This script loads the model from the configuration file and trains the model using the specified device and checkpoint (if provided).
"""

from argparse import ArgumentParser
from image_captioning import AdaptiveAttentionImageCaptioning

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to the configuration file")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--ckpt", "-c", default=None, help="Path to the checkpoint file")

    args = parser.parse_args()

    model = AdaptiveAttentionImageCaptioning.from_yaml(args.config)
    model.train(device=args.device, checkpoint=args.ckpt)
