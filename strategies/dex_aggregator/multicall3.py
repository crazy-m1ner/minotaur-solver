"""Multicall3 batching for read-only quote fan-out.

Multicall3 (``0xcA11bde05977b3631167028862bE2a173976CA11``) is deployed
at the same address on every major EVM chain. We use it to collapse
N sequential ``eth_call`` round-trips for the K-bucket × N-adapter
quote ladder into a single RPC, then locally decode the results.

If the chain doesn't have Multicall3 (rare — e.g. a fresh Anvil with
no preloaded contracts), ``batch_quote_calls`` falls back to sequential
``eth_call`` so behaviour stays correct, just slower.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from eth_abi.abi import decode as abi_decode, encode as abi_encode
from eth_hash.auto import keccak

from strategies.dex_aggregator.dex_adapter import Quote, eth_call

if TYPE_CHECKING:
    from strategies.dex_aggregator.dex_adapter import DexAdapter

logger = logging.getLogger(__name__)


MULTICALL3_ADDRESS = "0xcA11bde05977b3631167028862bE2a173976CA11"

_MC3_AGGREGATE3 = keccak(b"aggregate3((address,bool,bytes)[])")[:4]


@dataclass
class QuoteCall:
    """One adapter-issued read call, batchable through Multicall3.

    The ``adapter`` reference lets the post-batch decode step dispatch
    back to the issuing adapter's ``decode_quote`` without the routing
    layer having to know about adapter internals. ``descriptor`` is the
    adapter's own private context.
    """
    adapter: "DexAdapter"
    target: str
    calldata: bytes
    descriptor: Any


def batch_eth_calls(
    w3: Any, calls: list[tuple[str, bytes]],
) -> list[bytes | None] | None:
    """Send a batch of read-only ``eth_call``s through Multicall3.

    Returns a list of return-data, with ``None`` in any slot whose
    sub-call reverted. Returns ``None`` *for the whole batch* if the
    multicall request itself fails — callers fall back to sequential.
    """
    if not calls or w3 is None:
        return [] if not calls else None
    mc_calls = [(target, True, data) for target, data in calls]
    try:
        encoded = abi_encode(["(address,bool,bytes)[]"], [mc_calls])
    except Exception as exc:
        logger.warning("multicall encoding failed: %s", exc)
        return None
    raw = eth_call(w3, MULTICALL3_ADDRESS, _MC3_AGGREGATE3 + encoded)
    if raw is None:
        return None
    try:
        (results,) = abi_decode(["(bool,bytes)[]"], raw)
    except Exception as exc:
        logger.warning("multicall decoding failed: %s", exc)
        return None
    return [
        bytes(return_data) if success and return_data else None
        for success, return_data in results
    ]


def batch_quote_calls(
    w3: Any, calls: list[QuoteCall],
) -> list[Quote | None]:
    """Multicall-batch every QuoteCall and decode the responses.

    Falls back to per-call sequential ``eth_call`` if Multicall3 isn't
    reachable on the configured RPC.
    """
    if not calls:
        return []
    raw_results = batch_eth_calls(w3, [(c.target, c.calldata) for c in calls])
    if raw_results is None:
        logger.debug("Multicall3 unavailable — falling back to sequential")
        raw_results = [eth_call(w3, c.target, c.calldata) for c in calls]
    out: list[Quote | None] = []
    for call, raw in zip(calls, raw_results):
        if raw is None:
            out.append(None)
            continue
        try:
            out.append(call.adapter.decode_quote(call, raw))
        except Exception as exc:
            logger.debug("decode_quote on %s failed: %s", call.adapter.name, exc)
            out.append(None)
    return out
