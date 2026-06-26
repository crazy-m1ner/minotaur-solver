"""Miner solver for Subnet 112 (Minotaur).

The validator harness imports ``SOLVER_CLASS`` from this module and
calls ``generate_plan(intent, state, snapshot)`` on every order.

This fork ships ``MultiVenueSplitSolver`` as the default — a strict
super-set of ``BaselineSwapSolver``:

  - Uniswap V3 via QuoterV2 (multi-fee-tier search, batched via Multicall3)
  - Uniswap V2 + SushiSwap on Ethereum (V2-fork adapter)
  - Curve StableSwap with on-chain MetaRegistry discovery + meta-pool
    support (``get_dy_underlying`` / ``exchange_underlying``)
  - Aerodrome v1 (Solidly fork) on Base — complements the upstream's
    Aerodrome Slipstream (CL) path
  - K-bucket split-route allocation via O(N·K²) DP, with a one-shot
    probe that skips the split ladder when single-route depth is plenty
  - Approval dedup across split legs + recipient = ``state.contract_address``
    (AppIntentBase pattern)

Non-swap intents (cross-chain, yield, …) fall through to ``BaselineSwapSolver``
unchanged — additive, not replacement.

The validator's screening pipeline builds this repo as a Docker image
``FROM ghcr.io/subnet112/solver-base:v1`` and runs the runner harness,
which loads ``SOLVER_CLASS`` from ``/app/solver/solver.py``.
"""

from __future__ import annotations

import logging
import os

from strategies.dex_aggregator.multi_venue_solver import MultiVenueSplitSolver
from minotaur_subnet.sdk.intent_solver import SolverMetadata

logger = logging.getLogger(__name__)


SOLVER_NAME = os.environ.get("MINOTAUR_SOLVER_NAME", "multi-venue-split")
SOLVER_VERSION = os.environ.get("MINOTAUR_SOLVER_VERSION", "1.0.0")
SOLVER_AUTHOR = os.environ.get("MINOTAUR_SOLVER_AUTHOR", "miner")


class MinerSolver(MultiVenueSplitSolver):
    """The shipped solver. Pure cosmetic subclass — all behaviour lives
    in ``MultiVenueSplitSolver``. Fork-and-tweak target if you want to
    add a venue, change the split tuning, or layer your own strategy
    on top.
    """

    def metadata(self) -> SolverMetadata:
        base = super().metadata()
        return SolverMetadata(
            name=SOLVER_NAME,
            version=SOLVER_VERSION,
            author=SOLVER_AUTHOR,
            description=base.description,
            supported_chains=base.supported_chains,
            supported_intent_types=base.supported_intent_types,
        )


SOLVER_CLASS = MinerSolver
