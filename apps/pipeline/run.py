from __future__ import annotations

import argparse

from refresh import refresh
from train_model import main as train_main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--train", action="store_true")
    args = p.parse_args()

    if not args.refresh and not args.train:
        args.refresh = True
        args.train = True

    if args.refresh:
        refresh()
    if args.train:
        train_main()


if __name__ == "__main__":
    main()
