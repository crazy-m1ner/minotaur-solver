"""Uniswap V2 fork adapter — one adapter, many routers.

Every Uniswap-V2-compatible fork (UniswapV2, SushiSwap, PancakeSwap, …)
shares the same router ABI: ``getAmountsOut(uint256, address[])`` for
quoting and ``swapExactTokensForTokens(...)`` for execution. We register
multiple routers per chain in ``V2_DEPLOYMENTS`` and let the routing
layer pick the highest-output candidate per trade.

The route_summary tags which fork won so plan metadata reveals it.
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
class V2Deployment:
    chain_id: int
    router: str
    weth: str
    label: str = "V2"     # appears in route_summary so we can tell forks apart


# Multiple V2-shape routers per chain. UniV2 + Sushi V2 on ETH for now;
# add PancakeSwap on BSC etc. by appending to the chain's list.
V2_DEPLOYMENTS: dict[int, list[V2Deployment]] = {
    1: [
        V2Deployment(
            chain_id=1, label="UniswapV2",
            router="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            weth="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        ),
        V2Deployment(
            chain_id=1, label="SushiV2",
            router="0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            weth="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        ),
    ],
}

_V2_GET_AMOUNTS_OUT = keccak(b"getAmountsOut(uint256,address[])")[:4]
_V2_SWAP_EXACT_TOKENS = keccak(
    b"swapExactTokensForTokens(uint256,uint256,address[],address,uint256)",
)[:4]


@dataclass
class _V2RouteOpaque:
    deployment: V2Deployment
    path: list[str]


@dataclass
class _V2CallDescriptor:
    deployment: V2Deployment
    path: list[str]


class UniswapV2ForkAdapter(DexAdapter):
    """Quotes every registered V2-shape router + direct & 2-hop-via-WETH path."""

    name = "uniswap-v2-fork"

    def supports_chain(self, chain_id: int) -> bool:
        return bool(V2_DEPLOYMENTS.get(chain_id))

    def prepare_quote_calls(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[QuoteCall]:
        deployments = V2_DEPLOYMENTS.get(chain_id) or []
        if not deployments or amount_in <= 0:
            return []
        calls: list[QuoteCall] = []
        for dep in deployments:
            paths: list[list[str]] = [[token_in, token_out]]
            if dep.weth.lower() not in (token_in.lower(), token_out.lower()):
                paths.append([token_in, dep.weth, token_out])
            for path in paths:
                encoded = abi_encode(["uint256", "address[]"], [amount_in, path])
                calls.append(QuoteCall(
                    adapter=self, target=dep.router,
                    calldata=_V2_GET_AMOUNTS_OUT + encoded,
                    descriptor=_V2CallDescriptor(deployment=dep, path=list(path)),
                ))
        return calls

    def decode_quote(self, call: QuoteCall, raw: bytes) -> Quote | None:
        desc: _V2CallDescriptor = call.descriptor
        try:
            (amounts,) = abi_decode(["uint256[]"], raw)
        except Exception:
            return None
        if not amounts or amounts[-1] <= 0:
            return None
        return Quote(
            dex=self.name, amount_out=int(amounts[-1]),
            gas_estimate=140_000 if len(desc.path) == 2 else 220_000,
            route_summary=f"{desc.deployment.label} {' → '.join(desc.path)}",
            opaque=_V2RouteOpaque(deployment=desc.deployment, path=list(desc.path)),
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
        opaque: _V2RouteOpaque = quote.opaque
        encoded = abi_encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_in, min_out, opaque.path, recipient, deadline],
        )
        return [Interaction(
            target=opaque.deployment.router, value="0",
            call_data="0x" + (_V2_SWAP_EXACT_TOKENS + encoded).hex(),
            chain_id=chain_id,
        )]
