"""``DexAdapter`` protocol — the polymorphic surface every venue plugs into.

Each DEX (Uniswap V3 QuoterV2, Uniswap V2 / Sushi V2 router, Curve, Aerodrome
Solidly v1, …) implements the same three methods:

    warm_up(w3, chain, tin, tout)        # optional — discovery RPC
    prepare_quote_calls(...)             # pure encoding, no I/O
    decode_quote(call, raw)              # pure decoding

A default ``quote()`` glues prepare + decode together with a per-call
``eth_call`` so single-shot use stays simple. Multi-venue routing
(``best_route``, ``build_quote_ladder``) prefers the batched path via
``multicall3.batch_quote_calls``.

The point of the protocol is that adding a venue is one new subclass —
``best_route`` / split routing work over the union without code changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from eth_abi.abi import encode as abi_encode
from eth_hash.auto import keccak

from minotaur_subnet.shared.types import Interaction

logger = logging.getLogger(__name__)


# ── Tiny shared RPC helper (kept here so every adapter has it) ────────
def eth_call(w3: Any, to: str, data: bytes) -> bytes | None:
    """Plain ``eth_call`` wrapper. Returns None on any failure."""
    try:
        return bytes(w3.eth.call({"to": to, "data": "0x" + data.hex()}))
    except Exception as exc:
        logger.debug("eth_call to %s failed: %s", to, exc)
        return None


_ALLOWANCE = keccak(b"allowance(address,address)")[:4]


def read_allowance(w3: Any, token: str, owner: str, spender: str) -> int:
    """Read an ERC-20 allowance. Returns 0 on any RPC failure — the
    caller will then issue an ``approve`` to be safe."""
    encoded = abi_encode(["address", "address"], [owner, spender])
    raw = eth_call(w3, token, _ALLOWANCE + encoded)
    if not raw or len(raw) < 32:
        return 0
    try:
        return int.from_bytes(raw[:32], "big")
    except Exception:
        return 0


@dataclass
class Quote:
    """Adapter-agnostic quote.

    ``opaque`` carries adapter-specific data (path, fee tiers, pool address,
    stable flag, …) that ``build_interactions`` later needs to assemble the
    swap calldata. Treat it as opaque outside the issuing adapter.
    """

    dex: str
    amount_out: int
    gas_estimate: int
    route_summary: str
    opaque: Any = None


# ``QuoteCall`` lives in multicall3.py to avoid a circular import between
# dex_adapter (which DexAdapter uses) and multicall3 (which references
# DexAdapter via the QuoteCall.adapter field). We import lazily inside
# methods that need it.


class DexAdapter(ABC):
    """A venue with a batch-quotable quoting + execution surface."""

    name: str = "<unset>"

    @abstractmethod
    def supports_chain(self, chain_id: int) -> bool: ...

    def warm_up(
        self, w3: Any, chain_id: int, token_in: str, token_out: str,
    ) -> None:
        """Optional discovery step — populates adapter-local caches.

        Called by the routing primitives once per (chain, tin, tout) before
        ``prepare_quote_calls``. Default: no-op (adapters with fixed venues
        like Uniswap don't need it).
        """
        return None

    @abstractmethod
    def prepare_quote_calls(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[Any]:   # actually list[QuoteCall] — see note above
        """Build every QuoteCall this venue wants to issue for the trade.

        MUST be pure (no I/O) — anything requiring RPC belongs in
        ``warm_up``. Multiple QuoteCalls compete; the highest output wins.
        """

    def cached_quotes(
        self,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> list[Quote]:
        """Optional: return pre-computed Quotes without further RPC.

        For adapters whose pool state was fully read during ``warm_up``
        and is amount-independent (e.g. slot0/liquidity/tick on V3-fork
        pools), this is where amount-dependent output is computed
        locally via on-chain-equivalent math (pool_math.compute_v3_output)
        and returned without another RPC round-trip.

        Adapters that quote via a Quoter contract (Uniswap V3 QuoterV2,
        Uniswap V2 ``getAmountsOut``, Curve ``get_dy``) keep returning
        ``[]`` here and use ``prepare_quote_calls`` — the framework calls
        both and merges the results.

        Default: empty list.
        """
        return []

    @abstractmethod
    def decode_quote(self, call: Any, raw: bytes) -> Quote | None:
        """Decode a QuoteCall response. MUST be pure."""

    @abstractmethod
    def build_interactions(
        self,
        quote: Quote,
        chain_id: int,
        amount_in: int,
        min_out: int,
        recipient: str,
        deadline: int,
    ) -> list[Interaction]: ...

    @abstractmethod
    def approval_target(self, quote: Quote, chain_id: int) -> str | None:
        """Address that needs an ERC-20 allowance for the swap, or None."""

    def quote(
        self,
        w3: Any,
        chain_id: int,
        token_in: str,
        token_out: str,
        amount_in: int,
        *,
        preferred_fee: int | None = None,
    ) -> Quote | None:
        """Convenience: single-shot quote that hits the RPC sequentially.

        For real solving, prefer ``multicall3.batch_quote_calls`` to
        amortise RPC latency across many candidate calls.
        """
        if not self.supports_chain(chain_id):
            return None
        try:
            self.warm_up(w3, chain_id, token_in, token_out)
        except Exception as exc:
            logger.debug("warm_up on %s failed: %s", self.name, exc)
        calls = self.prepare_quote_calls(
            chain_id, token_in, token_out, amount_in,
            preferred_fee=preferred_fee,
        )
        best: Quote | None = None
        for call in calls:
            raw = eth_call(w3, call.target, call.calldata)
            if raw is None:
                continue
            try:
                q = self.decode_quote(call, raw)
            except Exception:
                continue
            if q and (best is None or q.amount_out > best.amount_out):
                best = q
        return best
