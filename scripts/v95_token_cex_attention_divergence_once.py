from pressure_graph.reports.v95_token_cex_attention_divergence import (
    write_v95_token_cex_attention_divergence,
)


if __name__ == "__main__":
    outputs = write_v95_token_cex_attention_divergence()
    for name, path in outputs.items():
        print(f"{name}: {path}")
