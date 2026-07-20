from pressure_graph.reports.v128_tg1_p2_orthogonal_combination import (
    write_v128_tg1_p2_orthogonal_combination,
)


if __name__ == "__main__":
    for name, path in write_v128_tg1_p2_orthogonal_combination().items():
        print(f"{name}: {path}")
