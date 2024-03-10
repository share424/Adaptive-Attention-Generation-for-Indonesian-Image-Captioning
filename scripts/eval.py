"""
This script is used to evaluate the performance of an Adaptive Attention Image Captioning model on a given dataset split.

Usage:
    python eval.py <config> [--split <split>] [--device <device>] [--ckpt <checkpoint>]

Arguments:
    config (str): Path to the configuration file.
    
Options:
    --split (str): Split to evaluate. Default is 'test'.
    --device (str): Device to use. Default is 'cuda'.
    --ckpt (str): Path to the checkpoint file.

Example:
    python eval.py config.yaml --split val --device cuda:0 --ckpt model.pth

"""

from argparse import ArgumentParser
from image_captioning import AdaptiveAttentionImageCaptioning

# Rest of the code...
from argparse import ArgumentParser

from image_captioning import AdaptiveAttentionImageCaptioning


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to the configuration file")
    parser.add_argument("--split", default="test", help="Split to evaluate")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--ckpt", "-c", help="Path to the checkpoint file")
    args = parser.parse_args()

    model = AdaptiveAttentionImageCaptioning.from_yaml(args.config)
    model.load_checkpoint(args.ckpt)
    scores = model.evaluate(split=args.split, device=args.device)
    print(scores)
