"""Curve StableSwap adapter with on-chain MetaRegistry discovery.

Mechanics:

  - ``warm_up(w3, chain, tin, tout)`` queries Curve's MetaRegistry for
    every pool serving the pair (up to ``_MAX_DISCOVERED_POOLS``) and
    caches the resulting ``(pool, i, j, is_underlying)`` tuples per
    (chain, in, out).
  - Static 3pool entry stays as a safety net so the adapter still works
    when MetaRegistry is unreachable.
  - Both ``exchange`` (regular pools) and ``exchange_underlying``
    (meta-pools — e.g. 3pool LP + base coin) variants supported via the
    ``is_underlying`` flag returned by ``get_coin_indices``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eth_abi.abi import decode as abi_decode, encode as abi_encode
from eth_hash.auto import keccak

from minotaur_subnet.shared.types import Interaction

from strategies.dex_aggregator.dex_adapter import DexAdapter, Quote, eth_call
from strategies.dex_aggregator.multicall3 import QuoteCall, batch_eth_calls

logger = logging.getLogger(__name__)


# MetaRegistry on Ethereum mainnet — combines registries for all Curve
# pool types (StableSwap, Crypto, StableNG, ...).
_CURVE_META_REGISTRY = "0xF98B45FA17DE75FB1aD0e7aFD971b0ca00e379fC"

# Curve's AddressProvider — deployed at the same address on every chain
# Curve supports. Maps registry ids → contract addresses. Id 7 ==
# MetaRegistry. Used to discover Base/L2 registries without hardcoding
# (the addresses differ per chain and change as Curve redeploys).
_CURVE_ADDRESS_PROVIDER = "0x5ffe7FB82894076ECB99A30D6A32e969e6e35E98"
_CURVE_GET_ADDRESS = keccak(b"get_address(uint256)")[:4]
_CURVE_META_REGISTRY_ID = 7

CURVE_REGISTRY_BY_CHAIN: dict[int, str] = {1: _CURVE_META_REGISTRY}
# Chains where we'll resolve MetaRegistry at warm-up time via AddressProvider.
# Avoids hardcoding addresses that may shift across Curve redeployments.
_CURVE_RESOLVE_CHAINS: tuple[int, ...] = (8453,)   # Base

# Pool selectors.
_CURVE_GET_DY = keccak(b"get_dy(int128,int128,uint256)")[:4]
_CURVE_EXCHANGE = keccak(b"exchange(int128,int128,uint256,uint256)")[:4]
# Meta-pool underlying variants (same arg layout).
_CURVE_GET_DY_UNDERLYING = keccak(b"get_dy_underlying(int128,int128,uint256)")[:4]
_CURVE_EXCHANGE_UNDERLYING = keccak(
    b"exchange_underlying(int128,int128,uint256,uint256)",
)[:4]
# Registry selectors.
_CURVE_FIND_POOL_FOR_COINS_I = keccak(
    b"find_pool_for_coins(address,address,uint256)",
)[:4]
_CURVE_GET_COIN_INDICES = keccak(b"get_coin_indices(address,address,address)")[:4]


@dataclass(frozen=True)
class CurvePool:
    address: str
    label: str
    chain_id: int = 1


@dataclass(frozen=True)
class CurveRoute:
    """A (pool, i, j) routing target for a specific token pair.

    ``is_underlying`` True ⇒ swap requires ``exchange_underlying`` /
    ``get_dy_underlying`` (meta-pools); False ⇒ regular ``exchange``.
    """
    pool: CurvePool
    i: int
    j: int
    is_underlying: bool = False


# Static seed — guaranteed to work even without MetaRegistry. Discovery
# augments this at runtime.
CURVE_STATIC_POOLS: list[tuple[CurvePool, tuple[str, ...]]] = [
    (
        CurvePool(
            address="0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",
            label="3pool",
        ),
        (
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",   # DAI
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",   # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",   # USDT
        ),
    ),
    # tricrypto2 (USDT/WBTC/WETH) — the canonical WBTC↔stable & WBTC↔ETH
    # path on Curve. Opens routes baseline's V3-only ETH mainnet adapter
    # set doesn't reach (no Aerodrome on chain 1, no Curve in baseline).
    (
        CurvePool(
            address="0xD51a44d3FaE010294C616388b506AcdA1bfAAE46",
            label="tricrypto2",
        ),
        (
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",   # USDT
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",   # WBTC
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
        ),
    ),
    # stETH/ETH — the deepest WETH↔stETH route on-chain. Triggers when a
    # benchmark scenario routes through staked ETH derivatives.
    (
        CurvePool(
            address="0xDC24316b9AE028F1497c275EB9192a3Ea0f67022",
            label="stETH/ETH",
        ),
        (
            "0x0000000000000000000000000000000000000000",   # native ETH (sentinel)
            "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",   # stETH
        ),
    ),
    # frxETH/WETH — Frax's ETH derivative with deep Curve liquidity.
    (
        CurvePool(
            address="0x9c3B46C0Ceb5B9e304FCd6D88Fc50f7DD24B31Bc",
            label="frxETH/WETH",
        ),
        (
            "0x5E8422345238F34275888049021821E8E08CAa1f",   # frxETH
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",   # WETH
        ),
    ),
]


@dataclass
class _CurveCallDescriptor:
    route: CurveRoute


@dataclass
class _CurveRouteOpaque:
    route: CurveRoute


def _index_in_tokens(tokens: tuple[str, ...], token: str) -> int | None:
    t = token.lower()
    for idx, addr in enumerate(tokens):
        if addr.lower() == t:
            return idx
    return None


class CurveAdapter(DexAdapter):
    """Curve StableSwap with MetaRegistry discovery + meta-pool support."""

    name = "curve"

    # Cap on pools probed per pair — defensive limit on RPC fan-out.
    # Most pairs have ≤3 pools; raise if MetaRegistry says otherwise.
    _MAX_DISCOVERED_POOLS = 8

    def __init__(self) -> None:
        self._route_cache: dict[tuple[int, str, str], list[CurveRoute]] = {}

    def supports_chain(self, chain_id: int) -> bool:
        if chain_id in CURVE_REGISTRY_BY_CHAIN:
            return True
        if chain_id in _CURVE_RESOLVE_CHAINS:
            return True
        return any(p.chain_id == chain_id for p, _ in CURVE_STATIC_POOLS)

    def _resolve_registry(self, w3: Any, chain_id: int) -> str | None:
        """Cache + return the MetaRegistry address for a chain.

        Already-known chains hit a hardcoded entry; resolve-chains
        (Base, …) look it up via Curve's AddressProvider at warm-up
        and stash the result. Returns None if discovery fails (adapter
        then yields no quotes for that chain — never crashes).
        """
        if chain_id in CURVE_REGISTRY_BY_CHAIN:
            return CURVE_REGISTRY_BY_CHAIN[chain_id]
        if chain_id not in _CURVE_RESOLVE_CHAINS or w3 is None:
            return None
        # AddressProvider.get_address(7) → MetaRegistry. One-shot eth_call.
        encoded = abi_encode(["uint256"], [_CURVE_META_REGISTRY_ID])
        raw = eth_call(
            w3, _CURVE_ADDRESS_PROVIDER, _CURVE_GET_ADDRESS + encoded,
        )
        if not raw or len(raw) < 32:
            return None
        addr = "0x" + raw[12:32].hex()
        if int(addr, 16) == 0:
            return None
        # Cache so we don't re-resolve.
        CURVE_REGISTRY_BY_CHAIN[chain_id] = addr
        logger.info("Resolved Curve MetaRegistry on chain %d: %s", chain_id, addr)
        return addr

    def warm_up(
        self, w3: Any, chain_id: int, token_in: str, token_out: str,
    ) -> None:
        cache_key = (chain_id, token_in.lower(), token_out.lower())
        if cache_key in self._route_cache:
            return

        routes: list[CurveRoute] = []

        # Static seed first.
        for pool, tokens in CURVE_STATIC_POOLS:
            if pool.chain_id != chain_id:
                continue
            i = _index_in_tokens(tokens, token_in)
            j = _index_in_tokens(tokens, token_out)
            if i is not None and j is not None and i != j:
                routes.append(CurveRoute(pool=pool, i=i, j=j))

        # MetaRegistry discovery — Ethereum + any resolve-chain (Base, …).
        registry = self._resolve_registry(w3, chain_id)
        if registry and w3 is not None:
            discovered = self._discover_via_registry(
                w3, registry, chain_id, token_in, token_out,
                exclude={r.pool.address.lower() for r in routes},
            )
            routes.extend(discovered)

        self._route_cache[cache_key] = routes

    def _discover_via_registry(
        self, w3: Any, registry: str, chain_id: int,
        token_in: str, token_out: str, exclude: set[str],
    ) -> list[CurveRoute]:
        # Step 1: find_pool_for_coins(in, out, i) for i in 0..N-1.
        find_calls: list[tuple[str, bytes]] = []
        for idx in range(self._MAX_DISCOVERED_POOLS):
            encoded = abi_encode(
                ["address", "address", "uint256"],
                [token_in, token_out, idx],
            )
            find_calls.append((registry, _CURVE_FIND_POOL_FOR_COINS_I + encoded))

        find_results = batch_eth_calls(w3, find_calls)
        if find_results is None:
            find_results = [
                eth_call(w3, registry, _CURVE_FIND_POOL_FOR_COINS_I + abi_encode(
                    ["address", "address", "uint256"], [token_in, token_out, idx]
                ))
                for idx in range(self._MAX_DISCOVERED_POOLS)
            ]

        pool_addresses: list[str] = []
        for raw in find_results:
            if not raw or len(raw) < 32:
                break
            addr_bytes = raw[12:32]
            addr = "0x" + addr_bytes.hex()
            if int(addr, 16) == 0:
                break
            if addr.lower() in exclude or addr.lower() in (p.lower() for p in pool_addresses):
                continue
            pool_addresses.append(addr)

        if not pool_addresses:
            return []

        # Step 2: get_coin_indices(pool, in, out) for each.
        idx_calls: list[tuple[str, bytes]] = []
        for pool_addr in pool_addresses:
            encoded = abi_encode(
                ["address", "address", "address"],
                [pool_addr, token_in, token_out],
            )
            idx_calls.append((registry, _CURVE_GET_COIN_INDICES + encoded))
        idx_results = batch_eth_calls(w3, idx_calls)
        if idx_results is None:
            idx_results = [
                eth_call(w3, registry, _CURVE_GET_COIN_INDICES + abi_encode(
                    ["address", "address", "address"],
                    [pool_addr, token_in, token_out],
                ))
                for pool_addr in pool_addresses
            ]

        out: list[CurveRoute] = []
        for pool_addr, raw in zip(pool_addresses, idx_results):
            if not raw:
                continue
            try:
                i, j, is_underlying = abi_decode(["int128", "int128", "bool"], raw)
            except Exception:
                continue
            if i == j:
                continue
            out.append(CurveRoute(
                pool=CurvePool(
                    address=pool_addr,
                    label=f"pool {pool_addr[:8]}…",
                    chain_id=chain_id,
                ),
                i=int(i), j=int(j),
                is_underlying=bool(is_underlying),
            ))
        return out

    def _routes_for_pair(
        self, chain_id: int, token_in: str, token_out: str,
    ) -> list[CurveRoute]:
        return self._route_cache.get(
            (chain_id, token_in.lower(), token_out.lower()), [],
        )

    def prepare_quote_calls(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[QuoteCall]:
        if amount_in <= 0:
            return []
        out: list[QuoteCall] = []
        for route in self._routes_for_pair(chain_id, token_in, token_out):
            encoded = abi_encode(
                ["int128", "int128", "uint256"], [route.i, route.j, amount_in],
            )
            selector = _CURVE_GET_DY_UNDERLYING if route.is_underlying else _CURVE_GET_DY
            out.append(QuoteCall(
                adapter=self, target=route.pool.address,
                calldata=selector + encoded,
                descriptor=_CurveCallDescriptor(route=route),
            ))
        return out

    def decode_quote(self, call: QuoteCall, raw: bytes) -> Quote | None:
        desc: _CurveCallDescriptor = call.descriptor
        if not raw or len(raw) < 32:
            return None
        try:
            amount_out = int.from_bytes(raw[:32], "big")
        except Exception:
            return None
        if amount_out <= 0:
            return None
        return Quote(
            dex=self.name, amount_out=int(amount_out), gas_estimate=180_000,
            route_summary=f"Curve {desc.route.pool.label} idx{desc.route.i}→idx{desc.route.j}",
            opaque=_CurveRouteOpaque(route=desc.route),
        )

    def approval_target(self, quote: Quote, chain_id: int) -> str | None:
        return quote.opaque.route.pool.address

    def build_interactions(
        self,
        quote: Quote,
        chain_id: int,
        amount_in: int,
        min_out: int,
        recipient: str,
        deadline: int,
    ) -> list[Interaction]:
        # ``exchange`` / ``exchange_underlying`` send tokens to msg.sender
        # (no recipient param). Under the AppIntentBase proxy pattern that's
        # the ephemeral proxy, which is exactly where the app contract
        # expects the output to land before settlement — no extra transfer.
        route = quote.opaque.route
        encoded = abi_encode(
            ["int128", "int128", "uint256", "uint256"],
            [route.i, route.j, amount_in, min_out],
        )
        selector = (
            _CURVE_EXCHANGE_UNDERLYING if route.is_underlying else _CURVE_EXCHANGE
        )
        return [Interaction(
            target=route.pool.address, value="0",
            call_data="0x" + (selector + encoded).hex(),
            chain_id=chain_id,
        )]
