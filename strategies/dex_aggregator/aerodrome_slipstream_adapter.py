"""Aerodrome Slipstream adapter — concentrated-liquidity V3-fork on Base.

Complements the upstream ``aerodrome.py`` which covers Aerodrome Slipstream
direct pools. Adds:

  - **Direct-pair quoting** via cached slot0/liquidity (warm-up batched in
    one Multicall3, quotes locally via ``pool_math.compute_v3_output``).
  - **Multi-hop quoting** via WETH/USDC intermediaries. The split routing
    layer composes (token_in → mid → token_out) by chaining two cached
    pools, so we close the gap where the upstream baseline beats us on
    Base BTC/DAI pairs via its own slipstream multi-hop discovery.

Slipstream is a V3 fork with two ABI differences from Uniswap V3:

  - Factory: ``getPool(tokenA, tokenB, int24 tickSpacing)`` (V3 uses uint24 fee).
  - Router exactInput/exactInputSingle: take ``int24 tickSpacing`` where
    V3 takes ``uint24 fee``.

Pool state shape (``slot0``, ``liquidity``, ``fee``, ``token0``, ``token1``)
matches V3 verbatim, so ``pool_math.compute_v3_output`` works without
modification. All RPC happens in ``warm_up`` (one round for getPool, one
round for pool state) and quoting is pure local math thereafter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eth_abi.abi import decode as abi_decode, encode as abi_encode
from eth_hash.auto import keccak

from minotaur_subnet.shared.types import Interaction

from strategies.dex_aggregator.aerodrome import (
    AERODROME_SLIPSTREAM_FACTORY,
    AERODROME_SLIPSTREAM_ROUTER,
    AERODROME_TICK_SPACINGS,
    encode_exact_input as encode_slipstream_multihop_swap,
    encode_exact_input_single as encode_slipstream_swap,
    encode_path as encode_slipstream_path,
)
from strategies.dex_aggregator.dex_adapter import DexAdapter, Quote
from strategies.dex_aggregator.multicall3 import QuoteCall, batch_eth_calls
from strategies.dex_aggregator.pool_math import compute_v3_output

logger = logging.getLogger(__name__)


# Selectors — kept minimal.
_GET_POOL_SELECTOR = keccak(b"getPool(address,address,int24)")[:4]
_SLOT0_SELECTOR = keccak(b"slot0()")[:4]
_LIQUIDITY_SELECTOR = keccak(b"liquidity()")[:4]
_FEE_SELECTOR = keccak(b"fee()")[:4]
_TOKEN0_SELECTOR = keccak(b"token0()")[:4]

# Multi-hop intermediaries per chain. Slipstream concentrates on Base.
# Bridge-token set widened beyond WETH/USDC so exotic-token pairs (which
# the team is now testing against, after moving token discovery out of
# solver code) still find paths through the deepest CL liquidity.
_SLIPSTREAM_INTERMEDIARIES: dict[int, tuple[str, ...]] = {
    # Trimmed to WETH + USDC for benchmark-window budget.
    8453: (
        "0x4200000000000000000000000000000000000006",  # WETH
        "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
    ),
}


@dataclass
class _SlipstreamPool:
    address: str
    tick_spacing: int
    sqrt_price_x96: int
    liquidity: int
    fee_ppm: int
    token0: str        # canonical lowercase
    token1: str        # canonical lowercase


@dataclass
class _SlipstreamRouteOpaque:
    pool: _SlipstreamPool
    zero_for_one: bool


@dataclass
class _SlipstreamMultihopOpaque:
    """Two-pool multi-hop: in→mid→out via Slipstream CL pools."""
    pool_a: _SlipstreamPool          # in → mid
    pool_b: _SlipstreamPool          # mid → out
    zero_for_one_a: bool
    zero_for_one_b: bool
    path_tokens: list[str]           # [token_in, mid, token_out]


def _canon_pair(a: str, b: str) -> tuple[str, str]:
    """Canonical key (lower-sorted) so we cache by unordered pair."""
    a, b = a.lower(), b.lower()
    return (a, b) if a < b else (b, a)


class AerodromeSlipstreamAdapter(DexAdapter):
    """Slipstream CL — direct + multi-hop (via WETH/USDC) quoting.

    Pools discovered once per (chain, pair). The warm-up batches direct +
    every intermediary leg's getPool / slot0 / liquidity / fee / token0
    reads into two Multicall3 round-trips, then all quoting is local.
    """

    name = "aerodrome-slipstream"

    def __init__(self) -> None:
        # (chain_id, canon_pair) → list[_SlipstreamPool]
        self._pool_cache: dict[tuple[int, tuple[str, str]], list[_SlipstreamPool]] = {}

    def supports_chain(self, chain_id: int) -> bool:
        return chain_id in AERODROME_SLIPSTREAM_FACTORY

    # ── warm_up: discover direct + multi-hop pools in one batched pass ─
    def warm_up(
        self, w3: Any, chain_id: int, token_in: str, token_out: str,
    ) -> None:
        if w3 is None or chain_id not in AERODROME_SLIPSTREAM_FACTORY:
            return
        # Pairs we need: direct + (in, mid) + (mid, out) per intermediary.
        pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()

        def _add_pair(a: str, b: str) -> None:
            if a.lower() == b.lower():
                return
            key = _canon_pair(a, b)
            if key in seen_pairs:
                return
            seen_pairs.add(key)
            pairs.append((a, b))

        _add_pair(token_in, token_out)
        for mid in _SLIPSTREAM_INTERMEDIARIES.get(chain_id, ()):
            ml = mid.lower()
            if ml == token_in.lower() or ml == token_out.lower():
                continue
            _add_pair(token_in, mid)
            _add_pair(mid, token_out)

        # Skip pairs already cached.
        pairs_to_discover = [
            (a, b) for (a, b) in pairs
            if (chain_id, _canon_pair(a, b)) not in self._pool_cache
        ]
        if not pairs_to_discover:
            return

        factory = AERODROME_SLIPSTREAM_FACTORY[chain_id]

        # ── Round 1: getPool(tin, tout, ts) for every (pair, ts) ────────
        find_calls = []
        find_origins: list[tuple[tuple[str, str], int]] = []  # ((a,b), ts)
        for (a, b) in pairs_to_discover:
            for ts in AERODROME_TICK_SPACINGS:
                encoded = abi_encode(
                    ["address", "address", "int24"],
                    [a, b, int(ts)],
                )
                find_calls.append((factory, _GET_POOL_SELECTOR + encoded))
                find_origins.append(((a, b), ts))

        find_results = batch_eth_calls(w3, find_calls)
        if find_results is None:
            for (a, b) in pairs_to_discover:
                self._pool_cache[(chain_id, _canon_pair(a, b))] = []
            return

        # candidate_pools[pair] = list[(addr, ts)]
        candidate_pools: dict[tuple[str, str], list[tuple[str, int]]] = {
            _canon_pair(a, b): [] for (a, b) in pairs_to_discover
        }
        for raw, ((a, b), ts) in zip(find_results, find_origins):
            if not raw or len(raw) < 32:
                continue
            addr = "0x" + raw[12:32].hex()
            if int(addr, 16) == 0:
                continue
            candidate_pools[_canon_pair(a, b)].append((addr, ts))

        # ── Round 2: batch-read slot0 / liquidity / fee / token0 ────────
        all_state_calls = []
        state_origins: list[tuple[tuple[str, str], str, int]] = []  # ((a,b), addr, ts)
        for (a, b) in pairs_to_discover:
            key = _canon_pair(a, b)
            for (addr, ts) in candidate_pools[key]:
                all_state_calls.append((addr, _SLOT0_SELECTOR))
                all_state_calls.append((addr, _LIQUIDITY_SELECTOR))
                all_state_calls.append((addr, _FEE_SELECTOR))
                all_state_calls.append((addr, _TOKEN0_SELECTOR))
                state_origins.append((key, addr, ts))

        if not all_state_calls:
            for (a, b) in pairs_to_discover:
                self._pool_cache[(chain_id, _canon_pair(a, b))] = []
            return

        state_results = batch_eth_calls(w3, all_state_calls)
        if state_results is None:
            for (a, b) in pairs_to_discover:
                self._pool_cache[(chain_id, _canon_pair(a, b))] = []
            return

        pools_by_pair: dict[tuple[str, str], list[_SlipstreamPool]] = {
            _canon_pair(a, b): [] for (a, b) in pairs_to_discover
        }
        for i, (key, addr, ts) in enumerate(state_origins):
            slot0_raw = state_results[i * 4 + 0]
            liq_raw = state_results[i * 4 + 1]
            fee_raw = state_results[i * 4 + 2]
            t0_raw = state_results[i * 4 + 3]
            if not all([slot0_raw, liq_raw, fee_raw, t0_raw]):
                continue
            try:
                slot0 = abi_decode(
                    ["uint160", "int24", "uint16", "uint16", "uint16", "bool"],
                    slot0_raw,
                )
                sqrt_price_x96, _tick, *_ = slot0
                liquidity = int.from_bytes(liq_raw[:32], "big")
                fee_ppm = int.from_bytes(fee_raw[28:32], "big")
                token0 = ("0x" + t0_raw[12:32].hex()).lower()
            except Exception as exc:
                logger.debug(
                    "Slipstream state decode failed for %s (ts=%d): %s",
                    addr, ts, exc,
                )
                continue
            # token1 = the other side of the canonical pair.
            token1 = key[1] if token0 == key[0] else key[0]
            pools_by_pair[key].append(_SlipstreamPool(
                address=addr, tick_spacing=int(ts),
                sqrt_price_x96=int(sqrt_price_x96),
                liquidity=int(liquidity),
                fee_ppm=int(fee_ppm),
                token0=token0, token1=token1,
            ))

        for key, pools in pools_by_pair.items():
            self._pool_cache[(chain_id, key)] = pools

    # ── prepare_quote_calls: nothing, we use cached_quotes ────────────
    def prepare_quote_calls(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[QuoteCall]:
        return []

    def decode_quote(self, call: QuoteCall, raw: bytes) -> Quote | None:
        return None   # Never called — adapter has no RPC quotes.

    # ── cached_quotes: direct + multi-hop, all RPC-free ───────────────
    def cached_quotes(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[Quote]:
        if amount_in <= 0:
            return []
        out: list[Quote] = []
        tin_low = token_in.lower()
        tout_low = token_out.lower()

        # ── Direct pool quotes ──────────────────────────────────────
        direct_key = (chain_id, _canon_pair(token_in, token_out))
        for pool in self._pool_cache.get(direct_key, []):
            zero_for_one = (pool.token0 == tin_low)
            try:
                output = compute_v3_output(
                    sqrt_price_x96=pool.sqrt_price_x96,
                    liquidity=pool.liquidity,
                    amount_in=amount_in,
                    zero_for_one=zero_for_one,
                    fee_ppm=pool.fee_ppm,
                )
            except Exception as exc:
                logger.debug(
                    "Slipstream compute_v3_output (direct) failed pool=%s: %s",
                    pool.address[:10], exc,
                )
                continue
            if output <= 0:
                continue
            out.append(Quote(
                dex=self.name,
                amount_out=int(output),
                gas_estimate=170_000,
                route_summary=(
                    f"Slipstream ts={pool.tick_spacing} "
                    f"{token_in[:10]}→{token_out[:10]}"
                ),
                opaque=_SlipstreamRouteOpaque(
                    pool=pool, zero_for_one=zero_for_one,
                ),
            ))

        # ── Multi-hop quotes via WETH/USDC ──────────────────────────
        for mid in _SLIPSTREAM_INTERMEDIARIES.get(chain_id, ()):
            ml = mid.lower()
            if ml == tin_low or ml == tout_low:
                continue
            leg_a_pools = self._pool_cache.get(
                (chain_id, _canon_pair(token_in, mid)), [],
            )
            leg_b_pools = self._pool_cache.get(
                (chain_id, _canon_pair(mid, token_out)), [],
            )
            if not leg_a_pools or not leg_b_pools:
                continue
            for pool_a in leg_a_pools:
                z_a = (pool_a.token0 == tin_low)
                try:
                    mid_amount = compute_v3_output(
                        sqrt_price_x96=pool_a.sqrt_price_x96,
                        liquidity=pool_a.liquidity,
                        amount_in=amount_in,
                        zero_for_one=z_a,
                        fee_ppm=pool_a.fee_ppm,
                    )
                except Exception:
                    continue
                if mid_amount <= 0:
                    continue
                for pool_b in leg_b_pools:
                    z_b = (pool_b.token0 == ml)
                    try:
                        final_out = compute_v3_output(
                            sqrt_price_x96=pool_b.sqrt_price_x96,
                            liquidity=pool_b.liquidity,
                            amount_in=mid_amount,
                            zero_for_one=z_b,
                            fee_ppm=pool_b.fee_ppm,
                        )
                    except Exception:
                        continue
                    if final_out <= 0:
                        continue
                    out.append(Quote(
                        dex=self.name,
                        amount_out=int(final_out),
                        # Multi-hop V3-style swap is ~230k on Base — two
                        # pools, second slot warm.
                        gas_estimate=230_000,
                        route_summary=(
                            f"Slipstream "
                            f"ts={pool_a.tick_spacing}/{pool_b.tick_spacing} "
                            f"{token_in[:8]}→{mid[:8]}→{token_out[:8]}"
                        ),
                        opaque=_SlipstreamMultihopOpaque(
                            pool_a=pool_a, pool_b=pool_b,
                            zero_for_one_a=z_a, zero_for_one_b=z_b,
                            path_tokens=[token_in, mid, token_out],
                        ),
                    ))
        return out

    # ── execution ─────────────────────────────────────────────────────
    def approval_target(self, quote: Quote, chain_id: int) -> str | None:
        return AERODROME_SLIPSTREAM_ROUTER.get(chain_id)

    def build_interactions(
        self,
        quote: Quote,
        chain_id: int,
        amount_in: int,
        min_out: int,
        recipient: str,
        deadline: int,
    ) -> list[Interaction]:
        op = quote.opaque
        router = AERODROME_SLIPSTREAM_ROUTER[chain_id]
        if isinstance(op, _SlipstreamMultihopOpaque):
            # Slipstream router exactInput uses a packed path of
            # (token, tickSpacing, token, tickSpacing, token).
            tick_spacings = [op.pool_a.tick_spacing, op.pool_b.tick_spacing]
            path = encode_slipstream_path(op.path_tokens, tick_spacings)
            calldata = encode_slipstream_multihop_swap(
                path=path,
                recipient=recipient,
                deadline=deadline,
                amount_in=amount_in,
                amount_out_minimum=min_out,
            )
        else:
            assert isinstance(op, _SlipstreamRouteOpaque)
            pool = op.pool
            token_in_addr, token_out_addr = (
                (pool.token0, pool.token1) if op.zero_for_one
                else (pool.token1, pool.token0)
            )
            calldata = encode_slipstream_swap(
                token_in=token_in_addr,
                token_out=token_out_addr,
                tick_spacing=pool.tick_spacing,
                recipient=recipient,
                deadline=deadline,
                amount_in=amount_in,
                amount_out_minimum=min_out,
            )
        return [Interaction(
            target=router,
            value="0",
            call_data=calldata,
            chain_id=chain_id,
        )]
