from pressure_graph.reports.v163_within_bucket_liquidity_quality import (
    write_v163_within_bucket_liquidity_quality,
)


if __name__ == "__main__":
    for name, path in write_v163_within_bucket_liquidity_quality().items():
        print(f"{name}: {path}")
