"""
This script predicts a caption for a single image using the Adaptive Attention Image Captioning model.

Usage:
    python predict_single.py <config> [--ckpt <checkpoint>] [--image <image>] [--search-strategy <strategy>] [--beam-size <size>] [--visualize] [--device <device>]

Arguments:
    config              Path to the configuration file (required)
    --ckpt <checkpoint> Path to the checkpoint file
    --image <image>     Path to the image file
    --search-strategy <strategy>   Search strategy to use (default: beam_search)
    --beam-size <size>  Beam size (default: 5)
    --visualize         Visualize the attention
    --device <device>   Device to use (default: cuda)

"""
from argparse import ArgumentParser

from image_captioning import AdaptiveAttentionImageCaptioning


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to the configuration file")
    parser.add_argument("--ckpt", help="Path to the checkpoint file")
    parser.add_argument("--image", help="Path to the image file")
    parser.add_argument(
        "--search-strategy", "-s", 
        choices=["greedy_search", "beam_search"], 
        default="beam_search",
        help="Search strategy to use"
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--visualize", action="store_true", help="Visualize the attention")
    parser.add_argument("--device", default="cuda", help="Device to use")

    args = parser.parse_args()

    model = AdaptiveAttentionImageCaptioning.from_yaml(args.config)
    model.load_checkpoint(args.ckpt)

    output = model.predict_single(
        args.image,
        search_strategy=args.search_strategy,
        beam_size=args.beam_size,
        keep_special_token=args.visualize, # visualize need keep special token
        return_visualize=args.visualize,
    )

    if args.visualize:
        sentences, images = output
        print(sentences)

        images[0].show()
    else:
        print(output)