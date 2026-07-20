from pathlib import Path

from pressure_graph.binance_um_premium_history import (
    BinancePremiumConfig,
    write_premium_inventory,
)


if __name__ == "__main__":
    root = Path("data/external/binance_um_premium_15m")
    expected = []
    manifest_path = root / "manifest.csv"
    if manifest_path.exists():
        import pandas as pd

        expected = (
            pd.read_csv(manifest_path)["bybit_symbol"].dropna().astype(str).tolist()
        )
    inventory = write_premium_inventory(
        BinancePremiumConfig(output_root=root), expected_symbols=expected
    )
    print(inventory.to_string(index=False))
