
"""
This script creates a wordmap for Indonesian image captioning using COCO annotations.

Usage:
    python create_wordmap.py <annotations>... [--output <output>]

Arguments:
    annotations (List[str]): List of paths to COCO annotations.
    output (str): Path to the output wordmap JSON file.

Example:
    python create_wordmap.py annotations.json --output wordmap.json
"""

import json
from argparse import ArgumentParser

from image_captioning.tokenizer import create_wordmap


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("annotations", nargs="+", help="List of paths to COCO annotations")
    parser.add_argument("--output", help="Path to the output wordmap")

    args = parser.parse_args()

    wordmap = create_wordmap(args.annotations)
    with open(args.output, "w") as f:
        json.dump(wordmap, f)

