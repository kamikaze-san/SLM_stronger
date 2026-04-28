#!/usr/bin/env python3
"""Print the actual HuggingFace split layout for wics/strategy-qa."""

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("wics/strategy-qa")
    print(ds)
    print("\nSplit sizes:")
    for split, dataset in ds.items():
        print(f"- {split}: {len(dataset):,}")


if __name__ == "__main__":
    main()
