"""
This script splits a checkpoint file into separate encoder and decoder files.

Usage: python split_checkpoint.py <checkpoint> <output>

Arguments:
    checkpoint (str): Path to the checkpoint file.
    output (str): Path to the output directory.

The script loads the checkpoint file, extracts the encoder and decoder weights,
and saves them as separate files in the specified output directory.
"""

from argparse import ArgumentParser
import os

import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Path to the checkpoint file")
    parser.add_argument("output", help="Path to the output file")

    args = parser.parse_args()

    weights = torch.load(args.checkpoint, map_location="cpu")
    encoder = weights["encoder"]
    decoder = weights["decoder"]

    torch.save(
        encoder,
        os.path.join(args.output, "encoder.pth"),
    )
    torch.save(
        decoder,
        os.path.join(args.output, "decoder.pth"),
    )