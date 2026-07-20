from pressure_graph.reports.v167_alt_bucket_vol_front_btc_straddle import (
    write_v167_alt_bucket_vol_front_btc_straddle,
)


if __name__ == "__main__":
    for name, path in write_v167_alt_bucket_vol_front_btc_straddle().items():
        print(f"{name}: {path}")
