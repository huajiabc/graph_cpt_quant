from pressure_graph.reports.v168_quiet_front_rich_iv_short_straddle import (
    write_v168_quiet_front_rich_iv_short_straddle,
)


if __name__ == "__main__":
    for name, path in write_v168_quiet_front_rich_iv_short_straddle().items():
        print(f"{name}: {path}")
