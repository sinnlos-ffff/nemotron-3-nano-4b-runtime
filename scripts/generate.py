"""Stable CLI scaffold; model loading and decoding arrive in the final milestone."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True, help="Local Hugging Face checkpoint directory")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.parse_args()
    parser.exit(2, "Generation is pending the Mamba, hybrid assembly, and weight-loading milestones.\n")


if __name__ == "__main__":
    main()
