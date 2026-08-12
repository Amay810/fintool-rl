"""Framework-neutral controller for the pre-registered online graph curriculum."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def discount_from_kl(kl: float) -> float:
    """Frozen monotone engineering schedule; simulation calibration may revise before training."""
    if kl < 0 or not math.isfinite(kl):
        raise ValueError("KL must be finite and non-negative")
    if kl < 0.01:
        return 0.98
    if kl < 0.03:
        return 0.96
    if kl < 0.06:
        return 0.93
    return 0.90


@dataclass
class CellPosterior:
    cell: str
    stratum: str
    alpha: float = 1.0
    beta: float = 1.0
    expected_tokens: float = 1.0
    latest_policy_version: str = ""

    def update(
        self,
        successes: int,
        failures: int,
        *,
        sync_kl: float,
        policy_version: str,
        observed_tokens: float | None = None,
    ) -> None:
        if successes < 0 or failures < 0 or successes + failures == 0:
            raise ValueError("posterior update requires a non-empty non-negative group")
        gamma = discount_from_kl(sync_kl)
        self.alpha = gamma * self.alpha + successes
        self.beta = gamma * self.beta + failures
        if observed_tokens is not None:
            if observed_tokens <= 0:
                raise ValueError("observed token cost must be positive")
            self.expected_tokens = gamma * self.expected_tokens + (1.0 - gamma) * observed_tokens
        self.latest_policy_version = policy_version

    def expected_drgrpo_mass(self, group_size: int) -> float:
        if group_size < 2:
            raise ValueError("group size must be at least two")
        total = self.alpha + self.beta
        expected_p_one_minus_p = self.alpha * self.beta / (total * (total + 1.0))
        return 2.0 * (group_size - 1) * expected_p_one_minus_p

    def acquisition_score(self, group_size: int) -> float:
        return self.expected_drgrpo_mass(group_size) / self.expected_tokens


@dataclass(frozen=True)
class TokenQuota:
    stratum: str
    minimum_tokens: int


def choose_cell(
    posteriors: Iterable[CellPosterior],
    quotas: Iterable[TokenQuota],
    spent_tokens: dict[str, int],
    *,
    group_size: int,
) -> CellPosterior:
    """Select only inside the most underfilled stratum, then maximize mass/token."""
    posterior_list = list(posteriors)
    quota_list = list(quotas)
    if not posterior_list or not quota_list:
        raise ValueError("posteriors and quotas must be non-empty")
    deficits = {
        quota.stratum: quota.minimum_tokens - spent_tokens.get(quota.stratum, 0)
        for quota in quota_list
    }
    target_stratum = max(sorted(deficits), key=lambda name: deficits[name])
    candidates = [item for item in posterior_list if item.stratum == target_stratum]
    if not candidates:
        raise ValueError(f"no cell for required stratum {target_stratum}")
    return max(candidates, key=lambda item: (item.acquisition_score(group_size), item.cell))
