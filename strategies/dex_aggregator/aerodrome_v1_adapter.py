"""Aerodrome v1 (Solidly fork) adapter on Base.

Complements the upstream ``aerodrome.py`` which covers Aerodrome
**Slipstream** (concentrated-liquidity V3-style pools). This adapter
covers the original Aerodrome v1 pools — Solidly-style ``stable`` and
``volatile`` curves, ``Route = (from, to, stable, factory)``.

Both Aerodrome variants share the same router function names but with
different ``Route`` encoding, so we keep them in separate adapters to
avoid mixing the two ABI shapes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from eth_abi.abi import decode as abi_decode, encode as abi_encode
from eth_hash.auto import keccak

from minotaur_subnet.shared.types import Interaction

from strategies.dex_aggregator.dex_adapter import DexAdapter, Quote
from strategies.dex_aggregator.multicall3 import QuoteCall

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AerodromeV1Deployment:
    chain_id: int
    router: str
    factory: str
    weth: str
    intermediaries: tuple[str, ...]   # 2-hop probe set; first is wrapped native


AERODROME_V1_DEPLOYMENTS: dict[int, AerodromeV1Deployment] = {
    8453: AerodromeV1Deployment(
        chain_id=8453,
        router="0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43",
        factory="0x420DD381b31aEf6683db6B902084cB0FFECe40Da",
        weth="0x4200000000000000000000000000000000000006",
        # Trimmed to WETH + USDC only to bound benchmark RPC budget under
        # the round's ~5-min window. The wider intermediary set caused
        # benchmark_window_elapsed; the same coverage is reachable through
        # other adapters' split legs (Slipstream multi-hop, V3 2-hop).
        intermediaries=(
            "0x4200000000000000000000000000000000000006",   # WETH
            "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",   # USDC
        ),
    ),
}

_AERO_GET_AMOUNTS_OUT = keccak(
    b"getAmountsOut(uint256,(address,address,bool,address)[])",
)[:4]
_AERO_SWAP_EXACT_TOKENS = keccak(
    b"swapExactTokensForTokens(uint256,uint256,(address,address,bool,address)[],address,uint256)",
)[:4]


@dataclass(frozen=True)
class _AeroHop:
    """One hop of a Solidly route. ``stable`` picks the curve type."""
    token_in: str
    token_out: str
    stable: bool


@dataclass
class _AeroRouteOpaque:
    deployment: AerodromeV1Deployment
    hops: list[_AeroHop]


@dataclass
class _AeroCallDescriptor:
    deployment: AerodromeV1Deployment
    hops: list[_AeroHop]


def _routes_encoded(
    hops: list[_AeroHop], factory: str,
) -> list[tuple[str, str, bool, str]]:
    return [(h.token_in, h.token_out, h.stable, factory) for h in hops]


class AerodromeV1Adapter(DexAdapter):
    """Probes every (stable, volatile) combination for direct + 2-hop-via-WETH paths."""

    name = "aerodrome-v1"

    def supports_chain(self, chain_id: int) -> bool:
        return chain_id in AERODROME_V1_DEPLOYMENTS

    def prepare_quote_calls(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[QuoteCall]:
        dep = AERODROME_V1_DEPLOYMENTS.get(chain_id)
        if dep is None or amount_in <= 0:
            return []

        calls: list[QuoteCall] = []
        # Direct: token_in → token_out, stable + volatile.
        for stable in (False, True):
            hops = [_AeroHop(token_in, token_out, stable)]
            calls.append(self._build_call(amount_in, hops, dep))

        # 2-hop via each intermediary (skip when intermediary is on either side).
        tin_low, tout_low = token_in.lower(), token_out.lower()
        for mid in dep.intermediaries:
            if mid.lower() in (tin_low, tout_low):
                continue
            for stable_a in (False, True):
                for stable_b in (False, True):
                    hops = [
                        _AeroHop(token_in, mid, stable_a),
                        _AeroHop(mid, token_out, stable_b),
                    ]
                    calls.append(self._build_call(amount_in, hops, dep))
        return calls

    def _build_call(
        self, amount_in: int, hops: list[_AeroHop], dep: AerodromeV1Deployment,
    ) -> QuoteCall:
        routes = _routes_encoded(hops, dep.factory)
        encoded = abi_encode(
            ["uint256", "(address,address,bool,address)[]"],
            [amount_in, routes],
        )
        return QuoteCall(
            adapter=self, target=dep.router,
            calldata=_AERO_GET_AMOUNTS_OUT + encoded,
            descriptor=_AeroCallDescriptor(deployment=dep, hops=list(hops)),
        )

    def decode_quote(self, call: QuoteCall, raw: bytes) -> Quote | None:
        desc: _AeroCallDescriptor = call.descriptor
        try:
            (amounts,) = abi_decode(["uint256[]"], raw)
        except Exception:
            return None
        if not amounts or amounts[-1] <= 0:
            return None
        path_str = " → ".join([desc.hops[0].token_in] + [h.token_out for h in desc.hops])
        flags = "".join("S" if h.stable else "V" for h in desc.hops)
        return Quote(
            dex=self.name, amount_out=int(amounts[-1]),
            gas_estimate=160_000 if len(desc.hops) == 1 else 250_000,
            route_summary=f"Aerodrome[{flags}] {path_str}",
            opaque=_AeroRouteOpaque(deployment=desc.deployment, hops=list(desc.hops)),
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
        opaque: _AeroRouteOpaque = quote.opaque
        routes = _routes_encoded(opaque.hops, opaque.deployment.factory)
        encoded = abi_encode(
            ["uint256", "uint256", "(address,address,bool,address)[]",
             "address", "uint256"],
            [amount_in, min_out, routes, recipient, deadline],
        )
        return [Interaction(
            target=opaque.deployment.router, value="0",
            call_data="0x" + (_AERO_SWAP_EXACT_TOKENS + encoded).hex(),
            chain_id=chain_id,
        )]
