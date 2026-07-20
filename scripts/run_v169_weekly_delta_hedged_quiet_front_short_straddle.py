from pressure_graph.reports.v169_weekly_delta_hedged_quiet_front_short_straddle import (
    write_v169_weekly_delta_hedged_quiet_front_short_straddle,
)


if __name__ == "__main__":
    for name, path in write_v169_weekly_delta_hedged_quiet_front_short_straddle().items():
        print(f"{name}: {path}")
