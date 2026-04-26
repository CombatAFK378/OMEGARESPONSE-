# app/negotiation.py
# Broadcast and negotiation logic for OmegaResponse.
# Called by environment.py every 5 steps (NEGOTIATION_INTERVAL).
#
# Responsibilities:
#   1. Collect belief broadcasts from all 3 agents
#   2. Detect belief conflicts (agents disagree on priority zone)
#   3. Produce a negotiation summary visible to all agents
#   4. Track coordinated follow-up actions (for +0.05 dense reward)
#   5. Expose helpers used by main.py /broadcast endpoint

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from app.environment import EnvironmentState

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BeliefBroadcast:
    step: int
    agent: str
    message: str
    belief: dict        # free-form structured belief snapshot
    priority_zone: int  # agent's believed highest-priority zone (-1 if unknown)


@dataclass
class NegotiationSummary:
    step: int
    broadcasts: list[BeliefBroadcast]
    agreed_priority_zone: int           # -1 if no consensus
    conflict_detected: bool
    conflict_description: str
    coordination_score: float           # 0.0 - 1.0


# ---------------------------------------------------------------------------
# Belief extraction helpers
# ---------------------------------------------------------------------------

def _extract_priority_zone(belief: dict, message: str) -> int:
    """
    Extract the agent's priority zone from its belief dict or message text.
    Returns -1 if no zone can be determined.
    """
    # Check common belief keys
    for key in ("priority_zone", "zone_id", "cleared_zone", "target_zone"):
        if key in belief:
            try:
                return int(belief[key])
            except (ValueError, TypeError):
                pass

    # Scan message for "Zone N" or "zone_id=N"
    import re
    m = re.search(r"[Zz]one\s*[_]?(\d)", message)
    if m:
        return int(m.group(1))

    return -1


def _extract_broadcast_entries(
    broadcast_log: list[dict],
    since_step: int,
) -> list[BeliefBroadcast]:
    """
    Filter broadcast_log for entries at or after since_step,
    excluding negotiation_round meta-entries.
    """
    results = []
    for entry in broadcast_log:
        if entry.get("type") == "negotiation_round":
            continue
        if entry.get("step", 0) >= since_step:
            belief = entry.get("belief", {})
            message = entry.get("message", "")
            results.append(BeliefBroadcast(
                step=entry.get("step", 0),
                agent=entry.get("from", "unknown"),
                message=message,
                belief=belief,
                priority_zone=_extract_priority_zone(belief, message),
            ))
    return results


# ---------------------------------------------------------------------------
# Core negotiation logic
# ---------------------------------------------------------------------------

def run_negotiation_round(state: "EnvironmentState") -> NegotiationSummary:
    """
    Execute a negotiation round using the last 5 steps of broadcasts.
    Called by environment.py every NEGOTIATION_INTERVAL steps.

    Steps:
        1. Collect recent broadcasts from all agents
        2. Compare priority zones → detect conflict
        3. Compute coordination score
        4. Append negotiation summary to broadcast_log so all agents see it
        5. Return NegotiationSummary for reward logic

    Returns:
        NegotiationSummary with consensus/conflict status
    """
    since_step = max(0, state.step - 5)
    broadcasts = _extract_broadcast_entries(state.broadcast_log, since_step)

    # Group by agent — keep latest broadcast per agent
    latest_by_agent: dict[str, BeliefBroadcast] = {}
    for b in broadcasts:
        latest_by_agent[b.agent] = b

    # Extract priority zones that are actually stated (not -1)
    stated_zones = {
        agent: b.priority_zone
        for agent, b in latest_by_agent.items()
        if b.priority_zone >= 0
    }

    # Conflict = two or more agents disagree on priority zone
    unique_zones = set(stated_zones.values())
    conflict_detected = len(unique_zones) > 1 and len(stated_zones) >= 2

    # Agreed zone = zone stated by majority (or only zone if consensus)
    agreed_priority_zone = -1
    if unique_zones:
        from collections import Counter
        zone_counts = Counter(stated_zones.values())
        top_zone, top_count = zone_counts.most_common(1)[0]
        # Consensus if majority agree (or only one zone mentioned)
        if top_count >= len(stated_zones) // 2 + 1 or len(unique_zones) == 1:
            agreed_priority_zone = top_zone

    # Build conflict description
    if conflict_detected:
        parts = [f"{agent} → Zone {z}" for agent, z in stated_zones.items()]
        conflict_description = "Priority disagreement: " + " | ".join(parts)
    else:
        conflict_description = "Agents aligned" if agreed_priority_zone >= 0 else "No broadcasts received"

    # Coordination score: fraction of broadcasting agents that agree on priority
    if len(stated_zones) >= 2:
        from collections import Counter
        zone_counts = Counter(stated_zones.values())
        majority_count = zone_counts.most_common(1)[0][1]
        coordination_score = majority_count / len(stated_zones)
    elif len(stated_zones) == 1:
        coordination_score = 1.0  # only one agent broadcast, no conflict possible
    else:
        coordination_score = 0.5  # no data, neutral

    summary = NegotiationSummary(
        step=state.step,
        broadcasts=list(latest_by_agent.values()),
        agreed_priority_zone=agreed_priority_zone,
        conflict_detected=conflict_detected,
        conflict_description=conflict_description,
        coordination_score=round(coordination_score, 3),
    )

    # Append negotiation summary to broadcast_log so all agents can read it
    state.broadcast_log.append({
        "step": state.step,
        "type": "negotiation_round",
        "from": "NegotiationCoordinator",
        "message": conflict_description,
        "agreed_priority_zone": agreed_priority_zone,
        "conflict_detected": conflict_detected,
        "coordination_score": coordination_score,
        "agents_reported": list(latest_by_agent.keys()),
    })

    # If conflict detected, increment environment conflict score
    if conflict_detected:
        state.conflict_score += 1
        state.anomaly_log.append(
            f"step={state.step}: negotiation conflict — {conflict_description}"
        )

    return summary


# ---------------------------------------------------------------------------
# Coordinated follow-up detection
# ---------------------------------------------------------------------------

def check_coordinated_followup(
    state: "EnvironmentState",
    broadcast_agent: str,
    broadcast_step: int,
) -> bool:
    """
    Return True if an agent OTHER than broadcast_agent took an action
    within 2 steps after the broadcast — indicating coordination.
    Used to award the +0.05 broadcast coordination dense reward.
    """
    followup_window_end = broadcast_step + 2
    for action in state.action_history:
        if (
            action["agent"] != broadcast_agent
            and broadcast_step < action["step"] <= followup_window_end
        ):
            return True
    return False


def award_broadcast_coordination_rewards(state: "EnvironmentState") -> float:
    """
    Scan recent broadcasts and check for coordinated follow-ups.
    Award +0.05 for each qualifying broadcast.
    Called by environment.py at the end of each step.
    Returns total coordination reward for this step.
    """
    from app.rewards import dense_reward_broadcast_coordination  # local import — no circular dep

    total = 0.0
    recent_broadcasts = [
        b for b in state.broadcast_log
        if b.get("type") != "negotiation_round"
        and b.get("step", 0) == state.step - 1  # broadcasts from last step
    ]
    for b in recent_broadcasts:
        coordinated = check_coordinated_followup(
            state,
            broadcast_agent=b.get("from", ""),
            broadcast_step=b.get("step", 0),
        )
        if coordinated:
            state.coordinated_followups += 1
            total += dense_reward_broadcast_coordination(True)

    return round(total, 4)


# ---------------------------------------------------------------------------
# Negotiation state summary (for /state endpoint and demos)
# ---------------------------------------------------------------------------

def get_negotiation_status(state: "EnvironmentState") -> dict:
    """
    Return a human-readable summary of current negotiation state.
    Called by main.py /state endpoint via environment.get_full_state().
    """
    recent_neg_rounds = [
        b for b in state.broadcast_log
        if b.get("type") == "negotiation_round"
    ][-3:]

    return {
        "total_negotiation_rounds": len([
            b for b in state.broadcast_log
            if b.get("type") == "negotiation_round"
        ]),
        "coordinated_followups": state.coordinated_followups,
        "recent_rounds": recent_neg_rounds,
        "current_conflict_score": state.conflict_score,
        "command_mode": state.command_mode.value,
    }