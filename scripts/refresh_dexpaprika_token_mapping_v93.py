from __future__ import annotations

import argparse

from pressure_graph.reports.v93_token_mapping_live_coverage import write_token_mapping_live_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh conservative DexPaprika mappings for the live universe.")
    parser.add_argument("--symbols", default="", help="Optional comma-separated universe override.")
    parser.add_argument("--promote", action="store_true", help="Write newly qualified B mappings into the mapping catalog.")
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()] or None
    outputs = write_token_mapping_live_coverage(symbols=symbols, promote=args.promote)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
