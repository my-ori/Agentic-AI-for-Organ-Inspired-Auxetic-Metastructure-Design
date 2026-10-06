#!/usr/bin/env python3
"""
Generate logarithmically equally spaced epsilon values and save them to a text file.

Examples:
    python epsilon_dividing.py --range 0.3,3
    python epsilon_dividing.py --range 0.03,3 --output epsilon_values.txt
    python epsilon_dividing.py --range 0.03,3 --segments 20
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple


def parse_range(value: str) -> Tuple[float, float]:
    """Parse a range written as 'start,end'."""
    try:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError
        start, end = map(float, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Range must contain two numbers separated by a comma, "
            "for example: --range 0.3,3"
        ) from exc

    if not math.isfinite(start) or not math.isfinite(end):
        raise argparse.ArgumentTypeError("Both range values must be finite numbers.")
    if start <= 0 or end <= 0:
        raise argparse.ArgumentTypeError(
            "Both range values must be greater than zero."
        )
    if start == end:
        raise argparse.ArgumentTypeError(
            "The start and end values must be different."
        )

    return start, end


def logarithmic_values(
    start: float,
    end: float,
    segments: int = 10,
) -> list[float]:
    """
    Divide a positive range into logarithmically equal segments.

    Return ``segments`` values, excluding the start point and including
    the end point. Descending ranges are also supported.
    """
    if segments <= 0:
        raise ValueError("segments must be a positive integer")

    log_start = math.log(start)
    log_end = math.log(end)
    log_step = (log_end - log_start) / segments

    values = [
        math.exp(log_start + index * log_step)
        for index in range(1, segments + 1)
    ]

    # Ensure that the final value is exactly the user-provided endpoint.
    values[-1] = end
    return values


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate logarithmically equally spaced values, excluding the "
            "start point and including the end point, and save them to a file."
        )
    )
    parser.add_argument(
        "--range",
        dest="value_range",
        required=True,
        type=parse_range,
        metavar="START,END",
        help="Positive start and end values, for example: --range 0.3,3",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=10,
        help="Number of logarithmic segments; default: 10",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=10,
        help="Number of significant digits to write; default: 10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("10_epsion_values.txt"),
        metavar="FILE",
        help="Output text file; default: log_values.txt",
    )
    parser.add_argument(
        "--separator",
        choices=("newline", "comma", "space"),
        default="comma",
        help="Value separator in the text file; default: newline",
    )
    return parser


def main() -> int:
    """Run the command-line program."""
    parser = build_parser()
    args = parser.parse_args()

    if args.segments <= 0:
        parser.error("--segments must be greater than zero")
    if args.precision <= 0:
        parser.error("--precision must be greater than zero")

    start, end = args.value_range
    values = logarithmic_values(start, end, args.segments)
    formatted = [f"{value:.{args.precision}g}" for value in values]

    separators = {
        "newline": "\n",
        "comma": ",",
        "space": " ",
    }
    text = separators[args.separator].join(formatted) + "\n"

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"Error writing '{args.output}': {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(values)} values to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
