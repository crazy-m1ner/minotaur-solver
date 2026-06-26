"""Uniswap V3 adapter — QuoterV2-based.

Distinct from the upstream ``uniswap_v3.py`` / ``pool_math.py`` path which
computes output locally from ``slot0`` + ``liquidity`` + ``tick``. Both
approaches have merits:

  - **pool_math (upstream)**: zero extra RPC per quote, but re-implements
    V3 swap math; can drift from on-chain reality on tick-cross / price-
    limit edge cases.
  - **QuoterV2 (this adapter)**: one RPC per fee tier (or batched via
    Multicall3 — what we do), but uses Uniswap's official contract math,
    so it can't drift.

The two coexist — ``MultiVenueSplitSolver`` runs them in parallel through
the same routing layer and picks whichever quotes more at the trade's
exact size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eth_abi.abi import decode as abi_decode, encode as abi_encode
from eth_hash.auto import keccak

from minotaur_subnet.shared.types import Interaction

from strategies.dex_aggregator.dex_adapter import DexAdapter, Quote
from strategies.dex_aggregator.multicall3 import QuoteCall
from strategies.dex_aggregator.v3_codec import (
    EXACT_INPUT_SINGLE_SELECTOR_V1,
    EXACT_INPUT_SINGLE_SELECTOR_V2,
    encode_swap_path,
)

logger = logging.getLogger(__name__)

# ── ABI selectors ─────────────────────────────────────────────────────
_V3_QUOTE_EXACT_INPUT_SINGLE = keccak(
    b"quoteExactInputSingle((address,address,uint256,uint24,uint160))",
)[:4]
_V3_QUOTE_EXACT_INPUT = keccak(b"quoteExactInput(bytes,uint256)")[:4]

# Uniswap V3 SwapRouter exactInput (multi-hop path) selector — same on V1+V2.
_EXACT_INPUT_SELECTOR = keccak(b"exactInput((bytes,address,uint256,uint256,uint256))")[:4]


# ── Deployment registry ────────────────────────────────────────────────
@dataclass(frozen=True)
class V3Deployment:
    chain_id: int
    router: str          # SwapRouter (V1) or SwapRouter02 (V2)
    quoter: str          # QuoterV2
    weth: str            # Wrapped native (multi-hop intermediary)
    router_is_v2: bool   # SwapRouter02 omits deadline in exactInputSingle
    intermediaries: tuple[str, ...]   # 2-hop probe set; first should be WETH


V3_DEPLOYMENTS: dict[int, V3Deployment] = {
    1: V3Deployment(
        chain_id=1,
        router="0xE592427A0AEce92De3Edee1F18E0157C05861564",
        quoter="0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
        weth="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        router_is_v2=False,
        # Bridge-token intermediaries. Each adds 16 multi-hop probes (4 fee
        # tiers × 4 fee tiers); Multicall3 absorbs them all in one
        # round-trip. With the team moving token discovery out of solver
        # code, exotic-token pairs must route via deep bridge liquidity.
        intermediaries=(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",   # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",   # USDT
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",   # DAI
        ),
    ),
    8453: V3Deployment(
        chain_id=8453,
        router="0x2626664c2603336E57B271c5C0b26F421741e481",
        quoter="0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
        weth="0x4200000000000000000000000000000000000006",
        router_is_v2=True,
        intermediaries=(
            "0x4200000000000000000000000000000000000006",   # WETH
            "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",   # USDC
            "0xd9AAEc86B65D86f6A7B5B1b0c42FFA531710b6CA",   # USDbC (bridged)
            "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",   # cbBTC
        ),
    ),
}

V3_FEE_TIERS: tuple[int, ...] = (100, 500, 3000, 10000)


# Curated 3-hop intermediary pairs per chain. Format: each tuple is the
# ordered list of mid tokens (mid1, mid2) — adapter builds the path
# input → mid1 → mid2 → output and quotes at a single fee combo per hop
# (avoiding a 64-call combinatorial blow-up: 1 path per pair instead).
#
# Targeted at scenarios where 2-hop misses the better route — chiefly
# BTC/stablecoin paths on ETH (WBTC↔USDC tied at single-hop baseline
# until 3-hop opens up WBTC→WETH→USDT→USDC at the deepest tiers).
_V3_3HOP_INTERMEDIARY_PAIRS: dict[int, tuple[tuple[str, str], ...]] = {
    1: (
        # WBTC↔stable via WETH-USDT
        (
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",   # USDT
        ),
        # WBTC↔stable via WETH-USDC
        (
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",   # USDC
        ),
        # stable↔stable via WETH-USDT (sometimes outperforms single-hop)
        (
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",   # USDT
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
        ),
    ),
    8453: (
        # cbBTC↔stable via cbBTC-WETH-USDC (already exploited by Slipstream
        # multi-hop, but V3 fork has its own pool depths for these)
        (
            "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",   # cbBTC
            "0x4200000000000000000000000000000000000006",   # WETH
        ),
        # via WETH-USDC for exotic-token coverage
        (
            "0x4200000000000000000000000000000000000006",   # WETH
            "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",   # USDC
        ),
    ),
}

# Fee combos for 3-hop probing. Deeply-liquid pairs cluster at 500 bps,
# so probe (500, 500, 500) and (500, 500, 3000) — covers WBTC's 3000 tier
# fall-back. Keep narrow to bound the call budget.
_V3_3HOP_FEE_COMBOS: tuple[tuple[int, int, int], ...] = (
    (500, 500, 500),
    (3000, 500, 500),
    (500, 500, 3000),
    (3000, 500, 3000),
)


# ── Opaque payloads (private) ──────────────────────────────────────────
@dataclass
class _V3RouteOpaque:
    deployment: V3Deployment
    path_tokens: list[str]
    path_fees: list[int]


@dataclass
class _V3CallDescriptor:
    deployment: V3Deployment
    path_tokens: list[str]
    path_fees: list[int]
    is_single_hop: bool


class UniswapV3QuoterAdapter(DexAdapter):
    """Quotes every fee tier on direct single-hop, then 2-hop via each
    intermediary. Subclassable: pass a different ``deployments`` dict in
    __init__ to wire the same logic to Pancake V3, Sushi V3, etc.

    Per quote the adapter issues up to ``len(V3_FEE_TIERS)`` single-hop +
    ``len(intermediaries) × len(V3_FEE_TIERS) ** 2`` multi-hop QuoteCalls.
    The routing layer Multicall3-batches them across adapters in a single
    RPC.
    """

    name = "uniswap-v3-quoter"

    def __init__(
        self,
        deployments: dict[int, V3Deployment] | None = None,
    ) -> None:
        # Module-level default keeps backwards compat; passing an override
        # lets us instantiate the same adapter for V3 forks (Pancake, …).
        self._deployments = deployments if deployments is not None else V3_DEPLOYMENTS

    def supports_chain(self, chain_id: int) -> bool:
        return chain_id in self._deployments

    def prepare_quote_calls(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[QuoteCall]:
        dep = self._deployments.get(chain_id)
        if dep is None or amount_in <= 0:
            return []

        calls: list[QuoteCall] = []
        # Direct single-hop on every fee tier.
        for fee in V3_FEE_TIERS:
            encoded = abi_encode(
                ["(address,address,uint256,uint24,uint160)"],
                [(token_in, token_out, amount_in, fee, 0)],
            )
            calls.append(QuoteCall(
                adapter=self, target=dep.quoter,
                calldata=_V3_QUOTE_EXACT_INPUT_SINGLE + encoded,
                descriptor=_V3CallDescriptor(
                    deployment=dep,
                    path_tokens=[token_in, token_out],
                    path_fees=[fee],
                    is_single_hop=True,
                ),
            ))

        # 2-hop via each intermediary (skip when intermediary is on either side).
        tin_low, tout_low = token_in.lower(), token_out.lower()
        for mid in dep.intermediaries:
            if mid.lower() in (tin_low, tout_low):
                continue
            for fee_a in V3_FEE_TIERS:
                for fee_b in V3_FEE_TIERS:
                    tokens = [token_in, mid, token_out]
                    fees = [fee_a, fee_b]
                    path = encode_swap_path(tokens, fees)
                    encoded = abi_encode(["bytes", "uint256"], [path, amount_in])
                    calls.append(QuoteCall(
                        adapter=self, target=dep.quoter,
                        calldata=_V3_QUOTE_EXACT_INPUT + encoded,
                        descriptor=_V3CallDescriptor(
                            deployment=dep, path_tokens=tokens,
                            path_fees=fees, is_single_hop=False,
                        ),
                    ))

        # 3-hop via curated intermediary pairs. Limited to a handful of
        # fee combos so the call count stays bounded (4 combos × ~3 pairs
        # = ~12 extra calls per pair, vs the ~16 per 2-hop intermediary).
        # Targets BTC↔stable paths where 2-hop misses the deepest route.
        for mid1, mid2 in _V3_3HOP_INTERMEDIARY_PAIRS.get(chain_id, ()):
            m1l, m2l = mid1.lower(), mid2.lower()
            if m1l in (tin_low, tout_low) or m2l in (tin_low, tout_low):
                continue
            if m1l == m2l:
                continue
            for fee_a, fee_b, fee_c in _V3_3HOP_FEE_COMBOS:
                tokens = [token_in, mid1, mid2, token_out]
                fees = [fee_a, fee_b, fee_c]
                path = encode_swap_path(tokens, fees)
                encoded = abi_encode(["bytes", "uint256"], [path, amount_in])
                calls.append(QuoteCall(
                    adapter=self, target=dep.quoter,
                    calldata=_V3_QUOTE_EXACT_INPUT + encoded,
                    descriptor=_V3CallDescriptor(
                        deployment=dep, path_tokens=tokens,
                        path_fees=fees, is_single_hop=False,
                    ),
                ))
        return calls

    def decode_quote(self, call: QuoteCall, raw: bytes) -> Quote | None:
        desc: _V3CallDescriptor = call.descriptor
        if desc.is_single_hop:
            try:
                amount_out, _sqrt, _ticks, _gas = abi_decode(
                    ["uint256", "uint160", "uint32", "uint256"], raw,
                )
            except Exception:
                return None
        else:
            try:
                amount_out, _sqrts, _ticks, _gas = abi_decode(
                    ["uint256", "uint160[]", "uint32[]", "uint256"], raw,
                )
            except Exception:
                return None
        if amount_out <= 0:
            return None

        if desc.is_single_hop:
            summary = (
                f"V3 {desc.path_tokens[0]}→{desc.path_tokens[1]} "
                f"@ {desc.path_fees[0]/10000:.2f}%"
            )
            gas_est = 150_000
        else:
            summary = f"V3 {' → '.join(desc.path_tokens)} @ {[f/10000 for f in desc.path_fees]}"
            # 2-hop V3 swaps land ~170-200k gas on mainnet (warm slot after
            # first pool). 260k over-penalized us by ~90 bps on the JS gate.
            # 3-hop adds ~50k for the third pool slot.
            hop_count = len(desc.path_fees)
            gas_est = 200_000 if hop_count == 2 else 250_000

        return Quote(
            dex=self.name, amount_out=int(amount_out), gas_estimate=gas_est,
            route_summary=summary,
            opaque=_V3RouteOpaque(
                deployment=desc.deployment,
                path_tokens=list(desc.path_tokens),
                path_fees=list(desc.path_fees),
            ),
        )

    def approval_target(self, quote: Quote, chain_id: int) -> str | None:
        return quote.opaque.deployment.router

    def build_interactions(
        self,
        quote: Quote,
        chain_id: int,
        amount_in: int,
        min_out: int,
        recipient: str,
        deadline: int,
    ) -> list[Interaction]:
        opaque: _V3RouteOpaque = quote.opaque
        dep = opaque.deployment
        if len(opaque.path_tokens) == 2:
            ix = _build_exact_input_single(
                dep, opaque, amount_in, min_out, recipient, deadline, chain_id,
            )
        else:
            ix = _build_exact_input(
                dep, opaque, amount_in, min_out, recipient, deadline, chain_id,
            )
        return [ix]


def _build_exact_input_single(
    dep: V3Deployment, op: _V3RouteOpaque,
    amount_in: int, min_out: int, recipient: str, deadline: int, chain_id: int,
) -> Interaction:
    token_in, token_out = op.path_tokens
    if dep.router_is_v2:
        encoded = abi_encode(
            ["address", "address", "uint24", "address",
             "uint256", "uint256", "uint160"],
            [token_in, token_out, op.path_fees[0],
             recipient, amount_in, min_out, 0],
        )
        selector = EXACT_INPUT_SINGLE_SELECTOR_V2
    else:
        encoded = abi_encode(
            ["address", "address", "uint24", "address",
             "uint256", "uint256", "uint256", "uint160"],
            [token_in, token_out, op.path_fees[0],
             recipient, deadline, amount_in, min_out, 0],
        )
        selector = EXACT_INPUT_SINGLE_SELECTOR_V1
    return Interaction(
        target=dep.router, value="0",
        call_data="0x" + (selector + encoded).hex(),
        chain_id=chain_id,
    )


def _build_exact_input(
    dep: V3Deployment, op: _V3RouteOpaque,
    amount_in: int, min_out: int, recipient: str, deadline: int, chain_id: int,
) -> Interaction:
    path = encode_swap_path(op.path_tokens, op.path_fees)
    if dep.router_is_v2:
        encoded = abi_encode(
            ["bytes", "address", "uint256", "uint256"],
            [path, recipient, amount_in, min_out],
        )
    else:
        encoded = abi_encode(
            ["bytes", "address", "uint256", "uint256", "uint256"],
            [path, recipient, deadline, amount_in, min_out],
        )
    return Interaction(
        target=dep.router, value="0",
        call_data="0x" + (_EXACT_INPUT_SELECTOR + encoded).hex(),
        chain_id=chain_id,
    )


# ── PancakeSwap V3 — V3 fork on the same ABI ─────────────────────────
# Same fee tiers + identical QuoterV2 ABI; deployed on ETH + Base. The
# Pancake SmartRouter (their V3 router) omits ``deadline`` in
# exactInputSingle — V2-shape router. Quoter address is the same across
# ETH and Base.
PANCAKE_V3_DEPLOYMENTS: dict[int, V3Deployment] = {
    1: V3Deployment(
        chain_id=1,
        router="0x13f4EA83D0bd40E75C8222255bc855a974568Dd4",
        quoter="0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997",
        weth="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        router_is_v2=True,
        intermediaries=(
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",   # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",   # USDT
        ),
    ),
    8453: V3Deployment(
        chain_id=8453,
        router="0x1b81D678ffb9C0263b24A97847620C99d213eB14",
        quoter="0xB048Bbc1Ee6b733FFfCFb9e9CeF7375518e25997",
        weth="0x4200000000000000000000000000000000000006",
        router_is_v2=True,
        intermediaries=(
            "0x4200000000000000000000000000000000000006",   # WETH
            "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",   # USDC
        ),
    ),
}


class PancakeSwapV3QuoterAdapter(UniswapV3QuoterAdapter):
    """PancakeSwap V3 — same V3 quoting/encoding, different addresses.

    Liquidity is meaningful on Base for major pairs (WETH↔USDC, USDC↔USDT)
    and offers an alternative price source the Uniswap deployment can't
    match. Slots into the routing layer alongside Uniswap V3 with no
    further changes — both adapters' quotes feed the same QuoteLadder.
    """

    name = "pancakeswap-v3-quoter"

    def __init__(self) -> None:
        super().__init__(deployments=PANCAKE_V3_DEPLOYMENTS)
