"""Pure, label-safe bankroll and variance calculations for N13 Stage 4.

All financial calculations are restricted to ``favorite_final_win`` because
moneyline contracts settle on the final winner. N06's ``deficit_erased``
probability and conformal interval answer a different question and are rejected
at this boundary.
"""

from __future__ import annotations

from collections import defaultdict


WIN_MARKET_LABEL = "favorite_final_win"
RELIABILITY_FACTORS = {
    "reliable": 1.00,
    "thin": 0.50,
    "unreliable": 0.25,
    "unknown": 0.25,
    "n_a": 0.25,
}
VALID_TIERS = {1, 2, 3}


def expected_value(prob: float, decimal_odds: float, *, label: str) -> float:
    """Return expected net profit per dollar: ``p * decimal_odds - 1``.

    The required label is a code-level settlement guard, not metadata.
    """
    _assert_win_label(label)
    probability = _probability(prob, "prob")
    odds = _decimal_odds(decimal_odds)
    return probability * odds - 1.0


def kelly_fraction(prob: float, decimal_odds: float, *, label: str) -> float:
    """Return full Kelly for binary decimal odds, floored at zero.

    Formula: ``(p * d - 1) / (d - 1)`` where ``d`` is decimal odds.
    Full Kelly is shown for context only; Stage 4 suggests fractional Kelly.
    """
    _assert_win_label(label)
    probability = _probability(prob, "prob")
    odds = _decimal_odds(decimal_odds)
    return max(0.0, (probability * odds - 1.0) / (odds - 1.0))


def fractional_kelly(
    prob: float,
    decimal_odds: float,
    fraction: float,
    tier: int,
    conformal_width: float | None,
    reliability: str,
    *,
    label: str,
) -> float:
    """Return the Stage 4 suggested bankroll fraction before the hard cap.

    Policy formula: ``full_kelly * kelly_fraction * reliability_factor``.

    ``kelly_fraction`` defaults to 0.25 in runtime config. Quarter-Kelly is the
    sole parameter-estimation uncertainty adjustment and is deliberately
    conservative because no label-matched conformal interval exists for
    ``favorite_final_win``. There is no second uncertainty-width haircut;
    stacking one would double-count estimation risk.

    Tier factors are intentionally flat at 1.00. N05-N08 found baseline_C tied
    the full N06 model, so lower tiers are not penalized. ``conformal_width`` is
    accepted to keep the presentation contract explicit but does not alter the
    stake because N06's interval belongs to ``deficit_erased``.

    Reliability factors are explicit policy choices based on historical sample
    size: reliable=1.00, thin=0.50, unreliable/unknown=0.25. This is distinct
    from parameter uncertainty and therefore is not a duplicate Kelly haircut.
    """
    _assert_win_label(label)
    if int(tier) not in VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(VALID_TIERS)}, got {tier}")
    policy_fraction = _unit_interval(fraction, "fraction")
    if conformal_width is not None and not 0.0 <= float(conformal_width) <= 1.0:
        raise ValueError("conformal_width must be in [0,1] when supplied")
    factor = RELIABILITY_FACTORS.get(str(reliability).lower())
    if factor is None:
        raise ValueError(f"unsupported reliability={reliability!r}")
    return kelly_fraction(prob, decimal_odds, label=label) * policy_fraction * factor


def losing_streak_probability(
    win_prob: float,
    streak_length: int,
    n_bets: int,
    *,
    label: str,
) -> float:
    """Exact probability of at least one N-loss run in independent bets.

    A finite-state dynamic program tracks the current consecutive-loss run
    among paths that have not yet reached ``streak_length``. This avoids the
    common but incorrect approximation ``(1-p)**N`` for a whole season.
    """
    _assert_win_label(label)
    probability = _probability(win_prob, "win_prob")
    streak = _positive_int(streak_length, "streak_length")
    bets = _nonnegative_int(n_bets, "n_bets")
    if bets < streak:
        return 0.0
    surviving = [0.0] * streak
    surviving[0] = 1.0
    loss_prob = 1.0 - probability
    for _ in range(bets):
        next_state = [0.0] * streak
        next_state[0] = sum(surviving) * probability
        for run_length in range(streak - 1):
            next_state[run_length + 1] += surviving[run_length] * loss_prob
        surviving = next_state
    return min(1.0, max(0.0, 1.0 - sum(surviving)))


def expected_losing_streaks(
    win_prob: float,
    streak_length: int,
    n_bets: int,
    *,
    label: str,
) -> float:
    """Expected count of overlapping N-loss windows in independent bets.

    By linearity of expectation this is ``(n-N+1) * (1-p)**N``. Overlapping
    windows are counted because each is a distinct season interval.
    """
    _assert_win_label(label)
    probability = _probability(win_prob, "win_prob")
    streak = _positive_int(streak_length, "streak_length")
    bets = _nonnegative_int(n_bets, "n_bets")
    return max(0, bets - streak + 1) * (1.0 - probability) ** streak


def risk_of_ruin(
    bankroll: float,
    stake_fraction: float,
    win_prob: float,
    decimal_odds: float,
    n_bets: int,
    floor: float,
    *,
    label: str,
) -> float:
    """Exact finite-season probability of touching a drawdown floor.

    Assumptions are explicit policy choices: independent identically
    distributed bets, the same fraction of *current* bankroll rebalanced each
    bet, and an absorbing drawdown boundary. ``floor`` is a fraction of the
    starting bankroll (default runtime policy: 0.50), not literal zero; with
    fractional staking, exact zero is unreachable in finite time.

    Dynamic programming retains surviving path probability by win count. The
    bankroll at each state is the product of win and loss multipliers, while
    paths touching the configured floor are absorbed as ruin events.
    """
    _assert_win_label(label)
    starting_bankroll = _positive_float(bankroll, "bankroll")
    fraction = _unit_interval(stake_fraction, "stake_fraction", upper_open=True)
    probability = _probability(win_prob, "win_prob")
    odds = _decimal_odds(decimal_odds)
    bets = _nonnegative_int(n_bets, "n_bets")
    floor_fraction = _unit_interval(floor, "floor", upper_open=True)
    if fraction == 0.0 or bets == 0:
        return 0.0

    floor_amount = starting_bankroll * floor_fraction
    win_multiplier = 1.0 + fraction * (odds - 1.0)
    loss_multiplier = 1.0 - fraction
    surviving: dict[tuple[int, int], float] = {(0, 0): 1.0}
    ruined = 0.0
    for _ in range(bets):
        next_surviving: dict[tuple[int, int], float] = defaultdict(float)
        for (wins, losses), path_probability in surviving.items():
            for is_win, outcome_probability in ((True, probability), (False, 1.0 - probability)):
                next_wins = wins + int(is_win)
                next_losses = losses + int(not is_win)
                wealth = (
                    starting_bankroll
                    * win_multiplier**next_wins
                    * loss_multiplier**next_losses
                )
                branch_probability = path_probability * outcome_probability
                if wealth <= floor_amount + 1e-12:
                    ruined += branch_probability
                else:
                    next_surviving[(next_wins, next_losses)] += branch_probability
        surviving = dict(next_surviving)
    return min(1.0, max(0.0, ruined))


def comfort_stake_fraction(
    bankroll: float,
    proposed_fraction: float,
    win_prob: float,
    decimal_odds: float,
    n_bets: int,
    floor: float,
    comfort_threshold: float,
    *,
    label: str,
    iterations: int = 60,
) -> float:
    """Find the largest fraction at or below the proposal meeting ruin policy.

    Bisection is deterministic. The threshold and drawdown floor are user
    policy choices, not estimates produced by the research notebooks.
    """
    _assert_win_label(label)
    high = _unit_interval(proposed_fraction, "proposed_fraction", upper_open=True)
    threshold = _unit_interval(comfort_threshold, "comfort_threshold")
    if risk_of_ruin(
        bankroll, high, win_prob, decimal_odds, n_bets, floor, label=label
    ) <= threshold:
        return high
    low = 0.0
    for _ in range(_positive_int(iterations, "iterations")):
        midpoint = (low + high) / 2.0
        risk = risk_of_ruin(
            bankroll, midpoint, win_prob, decimal_odds, n_bets, floor, label=label
        )
        if risk <= threshold:
            low = midpoint
        else:
            high = midpoint
    return low


def _assert_win_label(label: str) -> None:
    # N09 exposed the false edge created by applying a deficit-erasure model
    # to a final-win bet. Stage 3 and Stage 4 repeat this guard at the market
    # and bankroll boundaries so that cross-label bug cannot recur silently.
    if label != WIN_MARKET_LABEL:
        raise ValueError(
            f"risk math target is {WIN_MARKET_LABEL!r}; refusing mismatched label {label!r}"
        )


def _probability(value: float, name: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be in (0,1), got {number}")
    return number


def _decimal_odds(value: float) -> float:
    number = float(value)
    if number <= 1.0:
        raise ValueError(f"decimal_odds must be >1, got {number}")
    return number


def _unit_interval(value: float, name: str, *, upper_open: bool = False) -> float:
    number = float(value)
    valid = 0.0 <= number < 1.0 if upper_open else 0.0 <= number <= 1.0
    if not valid:
        bracket = "[0,1)" if upper_open else "[0,1]"
        raise ValueError(f"{name} must be in {bracket}, got {number}")
    return number


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _nonnegative_int(value: int, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number
