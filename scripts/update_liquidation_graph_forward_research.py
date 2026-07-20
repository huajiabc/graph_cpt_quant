from __future__ import annotations

import json

from pressure_graph.okx_liquidation_forward import (
    collect_okx_liquidation_snapshot,
)
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)
from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    write_v2334_audit,
)
from pressure_graph.reports.v2342_liquidation_graph_hourly_forward_ledger import (
    write_v2342,
)


if __name__ == "__main__":
    manifest = collect_okx_liquidation_snapshot(["BTCUSDT", *FROZEN_SYMBOLS])
    audit_paths = write_v2334_audit()
    ledger_paths = write_v2342()
    print(
        json.dumps(
            {
                "successful_symbols": int(manifest["error"].isna().sum()),
                "new_liquidation_events": int(manifest["new_rows"].sum()),
                "total_liquidation_events": int(manifest["total_rows"].sum()),
                "audit": str(audit_paths["audit"]),
                "hourly_ledger": str(ledger_paths["ledger"]),
                "ledger_metadata": str(ledger_paths["metadata"]),
            },
            indent=2,
        )
    )
