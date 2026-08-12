import pytest

from fintool_rl.curriculum import CellPosterior, TokenQuota, choose_cell, discount_from_kl


def test_kl_discount_is_monotone_and_bounded():
    values = [discount_from_kl(value) for value in (0.0, 0.02, 0.04, 0.10)]
    assert values == sorted(values, reverse=True)
    assert min(values) == 0.90
    assert max(values) == 0.98


def test_beta_expected_mass_peaks_at_balanced_posterior():
    balanced = CellPosterior("balanced", "long", alpha=5, beta=5, expected_tokens=10)
    easy = CellPosterior("easy", "long", alpha=9, beta=1, expected_tokens=10)
    assert balanced.expected_drgrpo_mass(8) > easy.expected_drgrpo_mass(8)


def test_quota_forces_long_stratum_before_within_stratum_score():
    short = CellPosterior("short", "short", alpha=5, beta=5, expected_tokens=1)
    long = CellPosterior("long", "long", alpha=5, beta=5, expected_tokens=100)
    chosen = choose_cell(
        [short, long],
        [TokenQuota("short", 100), TokenQuota("long", 100)],
        {"short": 100, "long": 0},
        group_size=8,
    )
    assert chosen.cell == "long"


def test_update_discounts_stale_evidence():
    posterior = CellPosterior("cell", "long", alpha=100, beta=1)
    posterior.update(0, 8, sync_kl=0.10, policy_version="v2", observed_tokens=50)
    assert posterior.alpha == pytest.approx(90)
    assert posterior.beta == pytest.approx(8.9)
    assert posterior.latest_policy_version == "v2"
