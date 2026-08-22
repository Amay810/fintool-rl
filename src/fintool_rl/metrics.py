"""Canonical financial metric vocabulary shared by data and tool contracts."""

from __future__ import annotations


# The names in this mapping are the only financial metric names accepted by the
# public financial-statement tools. SEC taxonomy tags are kept here as well so
# the importer and the agent-visible contract cannot drift independently.
CANONICAL_METRIC_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "total_assets": ("Assets",),
    "total_liabilities": ("Liabilities",),
}

CANONICAL_FINANCIAL_METRICS: tuple[str, ...] = tuple(CANONICAL_METRIC_TAGS)
