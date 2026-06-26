"""Split routing across DEX adapters.

The lever this module pulls: AMM output is concave in input. On large
trades any single venue gets blown out; splitting the input across two
or more venues equalises marginal output and recovers significant bps.

Pipeline:

  1. ``build_quote_ladder``  — one Multicall3 batches the full
     K-bucket × N-adapter quote matrix in one RPC.
  2. ``QuoteLadder.best_at(K)``     — single-route winner (the kth-bucket
     row across adapters).
  3. ``QuoteLadder.best_at(1)``     — small-trade reference, used by the
     probe to decide whether the split DP is worth running.
  4. ``_best_split_shares``  — O(N·K²) DP over the ladder, returns the
     allocation that maximises total output.
  5. ``split_from_ladder``   — builds ``SplitLeg``s; requotes the largest
     leg at its true amount (one extra RPC) when the trade doesn't
     divide evenly.

The old workflow built three overlapping ladders (single, probe,
split). The shared ``QuoteLadder`` cuts that to one RPC + an optional
remainder requote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from strategies.dex_aggregator.dex_adapter import DexAdapter, Quote
from strategies.dex_aggregator.multicall3 import QuoteCall, batch_quote_calls

logger = logging.getLogger(__name__)


# ── Warm-up dispatcher ────────────────────────────────────────────────
def warm_up_all(
    adapters: list[DexAdapter], w3: Any, chain_id: int,
    token_in: str, token_out: str,
) -> list[DexAdapter]:
    """Run ``warm_up`` on every chain-supporting adapter. Returns the
    eligible subset (parallel to the ``QuoteLadder.rows`` order)."""
    eligible: list[DexAdapter] = []
    for adapter in adapters:
        if not adapter.supports_chain(chain_id):
            continue
        try:
            adapter.warm_up(w3, chain_id, token_in, token_out)
        except Exception as exc:
            logger.warning("warm_up on %s failed: %s", adapter.name, exc)
        eligible.append(adapter)
    return eligible


# ── Single-route convenience ──────────────────────────────────────────
def best_route(
    adapters: list[DexAdapter],
    w3: Any,
    chain_id: int,
    token_in: str,
    token_out: str,
    amount_in: int,
    *,
    preferred_fee: int | None = None,
) -> Quote | None:
    """Highest-output single-route quote across the adapter set.

    Batches all candidate calls through Multicall3 in one round-trip.
    """
    eligible = warm_up_all(adapters, w3, chain_id, token_in, token_out)
    all_calls: list[QuoteCall] = []
    cached: list[Quote] = []
    for adapter in eligible:
        try:
            all_calls.extend(adapter.prepare_quote_calls(
                chain_id, token_in, token_out, amount_in,
                preferred_fee=preferred_fee,
            ))
        except Exception as exc:
            logger.warning("prepare_quote_calls on %s failed: %s", adapter.name, exc)
        try:
            cached.extend(adapter.cached_quotes(
                chain_id, token_in, token_out, amount_in,
                preferred_fee=preferred_fee,
            ))
        except Exception as exc:
            logger.warning("cached_quotes on %s failed: %s", adapter.name, exc)
    quotes = batch_quote_calls(w3, all_calls) + cached
    best: Quote | None = None
    for q in quotes:
        if q and (best is None or q.amount_out > best.amount_out):
            best = q
    return best


# ── Quote ladder ──────────────────────────────────────────────────────
@dataclass
class QuoteLadder:
    """The shared K-bucket × N-adapter Quote matrix.

    Built once per ``generate_plan`` call, consumed by single-route
    selection (``best_at(K)``) and split DP (``_best_split_shares``).
    """

    rows: list[list[Quote | None]]                # rows[adapter_idx][k]
    eligible: list[DexAdapter]                    # parallel to rows
    bucket: int
    remainder: int
    granularity: int

    def best_at(self, k: int) -> Quote | None:
        best: Quote | None = None
        for row in self.rows:
            q = row[k]
            if q and (best is None or q.amount_out > best.amount_out):
                best = q
        return best

    @property
    def amount_in(self) -> int:
        return self.bucket * self.granularity + self.remainder


def build_quote_ladder(
    adapters: list[DexAdapter],
    w3: Any,
    chain_id: int,
    token_in: str,
    token_out: str,
    amount_in: int,
    *,
    granularity: int = 10,
    preferred_fee: int | None = None,
) -> QuoteLadder | None:
    """Warm up adapters once and batch the full K-bucket quote matrix
    into a single Multicall3 RPC.

    Returns ``None`` if no adapter supports the chain or ``amount_in``
    is too small to bucket. The returned ladder is the canonical source
    for single-route selection, split probing, and split DP — none of
    those operations need to hit RPC again.
    """
    if amount_in <= 0 or granularity < 1:
        return None

    eligible = warm_up_all(adapters, w3, chain_id, token_in, token_out)
    if not eligible:
        return None

    bucket = amount_in // granularity
    if bucket == 0:
        return None
    remainder = amount_in - bucket * granularity

    all_calls: list[QuoteCall] = []
    call_origin: list[tuple[int, int]] = []
    cached_per_bucket: list[tuple[int, int, Quote]] = []   # (adapter_idx, k, quote)
    for ai, adapter in enumerate(eligible):
        for k in range(1, granularity + 1):
            amt = bucket * k
            try:
                for c in adapter.prepare_quote_calls(
                    chain_id, token_in, token_out, amt,
                    preferred_fee=preferred_fee,
                ):
                    all_calls.append(c)
                    call_origin.append((ai, k))
            except Exception as exc:
                logger.warning("prepare_quote_calls (k=%d) on %s failed: %s",
                               k, adapter.name, exc)
            try:
                for q in adapter.cached_quotes(
                    chain_id, token_in, token_out, amt,
                    preferred_fee=preferred_fee,
                ):
                    cached_per_bucket.append((ai, k, q))
            except Exception as exc:
                logger.warning("cached_quotes (k=%d) on %s failed: %s",
                               k, adapter.name, exc)

    quotes = batch_quote_calls(w3, all_calls)

    rows: list[list[Quote | None]] = [
        [None] * (granularity + 1) for _ in eligible
    ]
    for (ai, k), q in zip(call_origin, quotes):
        if q is None:
            continue
        cur = rows[ai][k]
        if cur is None or q.amount_out > cur.amount_out:
            rows[ai][k] = q
    # Layer cached (non-RPC) quotes on top.
    for ai, k, q in cached_per_bucket:
        cur = rows[ai][k]
        if cur is None or q.amount_out > cur.amount_out:
            rows[ai][k] = q

    return QuoteLadder(
        rows=rows, eligible=eligible, bucket=bucket,
        remainder=remainder, granularity=granularity,
    )


# ── Split allocation (DP) ─────────────────────────────────────────────
@dataclass
class SplitLeg:
    adapter_name: str
    amount_in: int
    quote: Quote


@dataclass
class SplitRoute:
    legs: list[SplitLeg]
    total_out: int
    total_gas_estimate: int

    @property
    def is_split(self) -> bool:
        return len([l for l in self.legs if l.amount_in > 0]) > 1

    @property
    def summary(self) -> str:
        active = [l for l in self.legs if l.amount_in > 0]
        return " + ".join(f"{l.adapter_name}@{l.amount_in}" for l in active)


def _best_split_shares(
    rows: list[list[Quote | None]], granularity: int,
) -> tuple[int, list[int]] | None:
    """Allocate exactly ``granularity`` buckets across ``len(rows)`` adapters
    to maximise total output.

    DP over adapters: ``dp[i][k]`` = best output using first ``i`` adapters
    summing to ``k`` buckets. Time: O(N·K²) — replaces a brute-force
    composition scan that blew up past ~5 adapters.
    """
    n = len(rows)
    if n == 0 or granularity < 1:
        return None

    NEG = -1
    dp: list[list[int]] = [[NEG] * (granularity + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    parent: list[list[int]] = [[-1] * (granularity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        row = rows[i - 1]
        for k in range(granularity + 1):
            best = NEG
            best_share = -1
            for share in range(k + 1):
                prev = dp[i - 1][k - share]
                if prev == NEG:
                    continue
                if share == 0:
                    delta = 0
                else:
                    q = row[share]
                    if q is None:
                        continue
                    delta = q.amount_out
                candidate = prev + delta
                if candidate > best:
                    best = candidate
                    best_share = share
            dp[i][k] = best
            parent[i][k] = best_share

    if dp[n][granularity] < 0:
        return None

    shares = [0] * n
    k = granularity
    for i in range(n, 0, -1):
        s = parent[i][k]
        shares[i - 1] = s
        k -= s
    return dp[n][granularity], shares


def _refine_shares(
    rows: list[list[Quote | None]],
    shares: list[int],
    granularity: int,
) -> tuple[int, list[int]]:
    """Local-search refinement on the DP solution.

    The bucket-DP enforces share ∈ {0..granularity} integer increments — so
    for granularity=20 the leg sizes step in 5%. When one leg's pool is
    significantly steeper than its peer's, the optimal allocation can sit
    between buckets and the DP rounds it to the nearest discrete option.
    On tied / near-tied scenarios (USDC↔WETH medium/large where baseline now
    catches up after its rewrite) that 5% rounding is exactly the regime
    where a couple more bps lives.

    This pass tries every (i, j, ±1) bucket swap between adopted legs and
    accepts swaps that lift total_out. O(active_legs²) per outer iteration,
    converges quickly. Zero RPC — operates on the existing ladder rows.
    """
    shares = list(shares)
    # Precompute output for a given (adapter_idx, share). 0 share contributes 0.
    def row_out(ai: int, sh: int) -> int:
        if sh <= 0:
            return 0
        q = rows[ai][sh]
        return q.amount_out if q is not None else -1

    n = len(shares)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for di in (-1, +1):
                    ni, nj = shares[i] + di, shares[j] - di
                    if not (0 <= ni <= granularity and 0 <= nj <= granularity):
                        continue
                    # Quote must exist for any non-zero share we propose.
                    out_ni = row_out(i, ni)
                    out_nj = row_out(j, nj)
                    if out_ni < 0 or out_nj < 0:
                        continue
                    cur = row_out(i, shares[i]) + row_out(j, shares[j])
                    new = out_ni + out_nj
                    if new > cur:
                        shares[i], shares[j] = ni, nj
                        improved = True
    total = sum(row_out(ai, sh) for ai, sh in enumerate(shares))
    return total, shares


def split_from_ladder(
    ladder: QuoteLadder,
    w3: Any,
    chain_id: int,
    token_in: str,
    token_out: str,
    *,
    preferred_fee: int | None = None,
) -> SplitRoute | None:
    """Run the DP + remainder-requote on a precomputed ladder.

    Adds at most one extra RPC (the remainder requote) when ``amount_in``
    doesn't divide evenly by ``granularity``. Otherwise pure CPU.
    """
    solution = _best_split_shares(ladder.rows, ladder.granularity)
    if solution is None:
        return None
    _total, shares = solution
    # Local refinement pass — picks up the residual bps the bucket-DP rounded
    # off when leg concavities differ.
    _total, shares = _refine_shares(ladder.rows, shares, ladder.granularity)

    largest_adapter_idx = max(range(len(shares)), key=lambda i: shares[i])
    legs: list[SplitLeg] = []
    requote_leg_idx: int | None = None
    for ai, share in enumerate(shares):
        if share == 0:
            continue
        leg_amount = ladder.bucket * share + (
            ladder.remainder if ai == largest_adapter_idx else 0
        )
        legs.append(SplitLeg(
            adapter_name=ladder.eligible[ai].name,
            amount_in=leg_amount,
            quote=ladder.rows[ai][share],
        ))
        if ai == largest_adapter_idx and ladder.remainder > 0:
            requote_leg_idx = len(legs) - 1

    if requote_leg_idx is not None:
        target = legs[requote_leg_idx]
        target_adapter = next(
            a for a in ladder.eligible if a.name == target.adapter_name
        )
        requote_calls = target_adapter.prepare_quote_calls(
            chain_id, token_in, token_out, target.amount_in,
            preferred_fee=preferred_fee,
        )
        if requote_calls:
            requoted = batch_quote_calls(w3, requote_calls)
            best_q = max(
                (q for q in requoted if q is not None),
                key=lambda q: q.amount_out, default=None,
            )
            if best_q is not None:
                target.quote = best_q

    total_out = sum(l.quote.amount_out for l in legs)
    total_gas = sum(l.quote.gas_estimate for l in legs)
    return SplitRoute(legs=legs, total_out=total_out, total_gas_estimate=total_gas)
