"""
Two-step firing — the swarm collaboration / consensus plane.

Motivation (user): LLMs get better with collaboration. Before committing an answer, the
selected swarm experts SYNC with each other, then VOTE on a shared solution — boosting
coherence and potency.

Grounding (established multi-agent literature, general knowledge — not part of the
mid-2026 verified research pass): this combines two effects shown to help repeatedly:
  * Multi-agent debate / "society of minds" — agents exchange drafts and revise, improving
    factuality and reasoning over a single pass.
  * Self-consistency — sampling multiple candidates and majority-voting beats greedy decode
    on reasoning tasks.
We fuse them: a DRAFT round (diversity), a SYNC round (collaboration/revision), and a VOTE
round (aggregation).

Pipeline:
    drafts  = [expert.answer(prompt)            for each expert]        # Step 1: independent
    synced  = [expert.revise(prompt, peers)     for each expert]        # Sync: see peers, revise
    winner  = aggregate(votes over synced)                             # Step 2: vote

The aggregator is pluggable: "majority" (self-consistency style on a normalized answer),
"judge" (an expert/mothership scores each candidate), or "borda" (ranked vote).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


class SwarmExpert(Protocol):
    """Minimal interface the consensus plane needs from a swarm member."""
    id: str

    def answer(self, prompt: str, adapters: list) -> str: ...
    def revise(self, prompt: str, own_draft: str, peer_drafts: dict, adapters: list) -> str: ...
    def vote(self, prompt: str, candidates: list[str]) -> int: ...           # index it prefers
    def score(self, prompt: str, candidate: str) -> float: ...               # for judge/weighted


@dataclass
class FiringTrace:
    drafts: dict[str, str]
    synced: dict[str, str]
    votes: dict[str, int]
    tally: dict[int, float]
    winner: str
    winner_expert: str
    rounds: int = 1


def _format_peers(drafts: dict[str, str], exclude: str, max_chars: int = 1200) -> dict:
    return {eid: d[:max_chars] for eid, d in drafts.items() if eid != exclude}


def two_step_firing(
    experts: list[SwarmExpert],
    prompt: str,
    adapters: list,
    aggregate: str = "judge",
    sync_rounds: int = 1,
    weight_by_score: bool = True,
) -> FiringTrace:
    """Run draft -> (sync x sync_rounds) -> vote across the selected experts.

    `adapters` are the Ortho-LoRA skills already resolved for this turn (every expert
    fires with the same skill set; diversity comes from the experts/seed, not the skills).
    """
    if not experts:
        raise ValueError("two_step_firing needs at least one expert")

    # ---- Step 1: independent drafts ----
    drafts = {e.id: e.answer(prompt, adapters) for e in experts}

    # ---- Sync: experts revise after seeing peers (the collaboration round) ----
    synced = dict(drafts)
    for _ in range(max(0, sync_rounds)):
        nxt = {}
        for e in experts:
            peers = _format_peers(synced, exclude=e.id)
            nxt[e.id] = e.revise(prompt, synced[e.id], peers, adapters)
        synced = nxt

    candidates = list(synced.values())
    cand_owner = list(synced.keys())

    # ---- Step 2: vote ----
    votes: dict[str, int] = {}
    tally: Counter = Counter()
    if aggregate == "majority":
        # self-consistency: normalize answers, count identical clusters
        norm = [c.strip().lower() for c in candidates]
        for eid in synced:                       # each expert "votes" for its own synced answer
            votes[eid] = norm.index(synced[eid].strip().lower())
        for v in votes.values():
            tally[v] += 1.0
    elif aggregate == "borda":
        for e in experts:
            pref = e.vote(prompt, candidates)    # top choice; extend to full ranking if available
            votes[e.id] = pref
            tally[pref] += 1.0
    else:  # "judge": each expert scores every candidate; sum (optionally self-weighted)
        for e in experts:
            best_idx, best_val = 0, float("-inf")
            for i, c in enumerate(candidates):
                s = e.score(prompt, c)
                w = e.score(prompt, synced[e.id]) if weight_by_score else 1.0
                tally[i] += s * (w if weight_by_score else 1.0)
                if s > best_val:
                    best_idx, best_val = i, s
            votes[e.id] = best_idx

    winner_idx = max(tally.items(), key=lambda kv: kv[1])[0]
    return FiringTrace(
        drafts=drafts, synced=synced, votes=votes, tally=dict(tally),
        winner=candidates[winner_idx], winner_expert=cand_owner[winner_idx],
        rounds=sync_rounds,
    )
