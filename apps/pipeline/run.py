from __future__ import annotations

import argparse

from refresh import refresh
from train_model import main as train_main
from infer_model import main as infer_main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--infer", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all:
        args.refresh = True
        args.train = True
        args.infer = False

    if not (args.refresh or args.train or args.infer or args.all):
        raise SystemExit("Use --refresh / --train / --infer / --all")

    if args.refresh:
        refresh()
    if args.train:
        train_main()
    if args.infer:
        infer_main()


if __name__ == "__main__":
    main()
