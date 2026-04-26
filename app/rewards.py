# app/rewards.py
# All 3 reward layers for OmegaResponse.
# MUST be importable standalone — GRPOTrainer calls compute_grpo_reward() directly.
# No imports from environment.py at module level (only inside TYPE_CHECKING block).
#
# GRPOTrainer reward function signature (HF TRL):
#   def reward_func(completions: list[str], **kwargs) -> list[float]
#   prompts are passed via kwargs.get("prompts", ...)

from __future__ import annotations
from typing import TYPE_CHECKING
import re

if TYPE_CHECKING:
    from app.environment import EnvironmentState


# ---------------------------------------------------------------------------
# Constants (duplicated here so rewards.py is fully standalone)
# ---------------------------------------------------------------------------

NUM_ZONES = 5
CONFLICT_SCORE_THRESHOLD = 2

# Layer 1 — Dense
R_AMBULANCE_REACHES        =  0.15
R_ROAD_CLEARED_BONUS       =  0.10
R_CONFLICT_PENALTY         = -0.10
R_MONITOR_MISS_PENALTY     = -0.05
R_BROADCAST_COORDINATION   =  0.05
R_CORRECT_MODE_SWITCH      =  0.08

# Layer 2 — Milestone
R_ZONE_STABILIZED                  =  0.30
R_HOSPITAL_ABOVE_50                =  0.20
R_ZONE_DETERIORATED                = -0.20
R_INTERVENTION_PREVENTED_CONFLICT  =  0.15

# Layer 3 — Terminal
R_ZONE_FULLY_RECOVERED       =  1.00
R_PREVENTABLE_CASUALTY       = -0.50
R_COORDINATION_EFFICIENCY_MAX =  0.50
R_OVERSIGHT_SCORE_MAX        =  0.30


# ---------------------------------------------------------------------------
# Layer 1 — Dense reward helpers (called from environment.py action handlers)
# ---------------------------------------------------------------------------

def dense_reward_ambulance_dispatched(road_cleared_first: bool) -> float:
    reward = R_AMBULANCE_REACHES
    if road_cleared_first:
        reward += R_ROAD_CLEARED_BONUS
    return reward


def dense_reward_conflict(is_conflict: bool) -> float:
    return R_CONFLICT_PENALTY if is_conflict else 0.0


def dense_reward_monitor_miss(monitor_missed: bool) -> float:
    return R_MONITOR_MISS_PENALTY if monitor_missed else 0.0


def dense_reward_broadcast_coordination(coordinated: bool) -> float:
    return R_BROADCAST_COORDINATION if coordinated else 0.0


def dense_reward_mode_switch(switched_correctly: bool) -> float:
    return R_CORRECT_MODE_SWITCH if switched_correctly else 0.0


# ---------------------------------------------------------------------------
# Layer 2 — Milestone reward (called by environment.py every 20 steps)
# ---------------------------------------------------------------------------

def compute_milestone_reward(state: "EnvironmentState") -> float:
    reward = 0.0

    for zone in state.zones:
        if zone.severity < 3.0:
            reward += R_ZONE_STABILIZED

        if zone.severity > 6.0 and zone.idle_resources_steps >= 5:
            reward += R_ZONE_DETERIORATED

    if state.hospital_capacity > 0.5:
        reward += R_HOSPITAL_ABOVE_50

    if state.interventions_correct > 0 and state.conflict_score == 0:
        reward += R_INTERVENTION_PREVENTED_CONFLICT

    return round(reward, 4)


# ---------------------------------------------------------------------------
# Layer 3 — Terminal reward (called by environment.py at step 300)
# ---------------------------------------------------------------------------

def compute_terminal_reward(state: "EnvironmentState") -> float:
    reward = 0.0

    # +1.00 per zone fully recovered
    recovered_zones = sum(1 for z in state.zones if z.severity == 0.0)
    reward += recovered_zones * R_ZONE_FULLY_RECOVERED

    # -0.50 per 10 preventable casualties (resources nearby but died)
    preventable = sum(
        z.casualties for z in state.zones
        if z.casualties > 0 and z.resources_available
    )
    reward += (preventable / 10.0) * R_PREVENTABLE_CASUALTY

    # +0.50 * coordination efficiency
    if state.total_actions > 0:
        efficiency = max(0.0, 1.0 - state.total_conflicts / state.total_actions)
        reward += efficiency * R_COORDINATION_EFFICIENCY_MAX

    # +0.30 * oversight score
    if state.interventions_total > 0:
        oversight = state.interventions_correct / state.interventions_total
        reward += oversight * R_OVERSIGHT_SCORE_MAX

    return round(reward, 4)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_action_from_completion(completion: str) -> tuple[str, dict]:
    """
    Parse action name and parameters from model completion text.
    Expected format (flexible):
        Action: DispatchAmbulance
        Parameters: {"zone_id": 2}
    """
    action = "NoOp"
    params: dict = {}

    # Action name
    action_match = re.search(
        r"[Aa]ction\s*[:\-]\s*([A-Za-z]+(?:[A-Z][a-z]+)*)",
        completion
    )
    if action_match:
        action = action_match.group(1).strip()

    # Structured zone_id
    zone_match = re.search(r"zone_id[\"']?\s*[:\-]\s*(\d)", completion)
    if zone_match:
        params["zone_id"] = int(zone_match.group(1))
    else:
        # Natural language fallback — catches "Zone 3", "zone 3", "zone3"
        z2 = re.search(r'\bzone\s*(\d)\b', completion, re.IGNORECASE)
        if z2:
            params["zone_id"] = int(z2.group(1))

    # vehicle_id
    vehicle_match = re.search(r"vehicle_id[\"']?\s*[:\-]\s*(\d)", completion)
    if vehicle_match:
        params["vehicle_id"] = int(vehicle_match.group(1))

    params.setdefault("zone_id", 0)
    return action, params


def _parse_state_from_prompt(prompt: str) -> dict:
    """
    Extract key state variables from the prompt string.
    Prompts are structured text blocks generated by rule_based_agent.py.
    """
    state: dict = {
        "zone_severities": {},
        "road_conditions": {},
        "hospital_capacity": 0.7,
        "conflict_score": 0,
        "command_mode": "MonitorMode",
        "agent_role": "MedAgent",
    }

    # Agent role
    role_match = re.search(r"You are the\s+(\w+)", prompt)
    if role_match:
        state["agent_role"] = role_match.group(1)

    # Zone severities: "Zone 0: severity=7.2"
    for m in re.finditer(r"Zone\s+(\d)\s*[:\|]\s*severity\s*=\s*([\d.]+)", prompt):
        state["zone_severities"][int(m.group(1))] = float(m.group(2))

    # Road conditions: "Zone 0 road: blocked"
    for m in re.finditer(r"Zone\s+(\d)\s*road\s*[:\|]\s*(\w+)", prompt, re.IGNORECASE):
        state["road_conditions"][int(m.group(1))] = m.group(2)

    # Hospital capacity
    hcap = re.search(r"hospital_capacity\s*[=:]\s*([\d.]+)", prompt)
    if hcap:
        state["hospital_capacity"] = float(hcap.group(1))

    # Conflict score
    cs = re.search(r"conflict_score\s*[=:]\s*(\d+)", prompt)
    if cs:
        state["conflict_score"] = int(cs.group(1))

    # Command mode
    if "InterventionMode" in prompt:
        state["command_mode"] = "InterventionMode"

    return state


def _heuristic_reward(action: str, params: dict, state: dict) -> float:
    """
    Fast heuristic reward scorer for GRPOTrainer.
    No env needed — uses parsed state from prompt string.
    """
    reward = 0.0
    zone_id = params.get("zone_id", 0)
    severities = state.get("zone_severities", {})
    road_conditions = state.get("road_conditions", {})
    conflict_score = state.get("conflict_score", 0)
    command_mode = state.get("command_mode", "MonitorMode")

    # MedAgent
    if action == "DispatchAmbulance":
        sev = severities.get(zone_id, 5.0)
        reward += (sev / 10.0) * R_AMBULANCE_REACHES
        if road_conditions.get(zone_id, "blocked") == "clear":
            reward += R_ROAD_CLEARED_BONUS

    elif action == "SetupFieldHospital":
        hcap = state.get("hospital_capacity", 0.7)
        reward += 0.08 if hcap < 0.5 else 0.02

    elif action == "TriageZone":
        sev = severities.get(zone_id, 5.0)
        reward += (sev / 10.0) * 0.06

    elif action in ("BroadcastMedUpdate", "BroadcastLogistUpdate"):
        reward += R_BROADCAST_COORDINATION

    elif action == "RequestBloodSupply":
        reward += 0.03

    # LogistAgent
    elif action == "ClearRoad":
        road = road_conditions.get(zone_id, "clear")
        reward += 0.07 if road in ("blocked", "damaged") else -0.02

    elif action == "RouteTruck":
        sev = severities.get(zone_id, 5.0)
        reward += (sev / 10.0) * 0.05

    elif action == "RequestAirDrop":
        sev = severities.get(zone_id, 5.0)
        reward += (sev / 10.0) * 0.08

    elif action == "RefuelVehicle":
        reward += 0.02

    # CommandAgent
    elif action == "OverrideAgent":
        if command_mode == "InterventionMode" and conflict_score > CONFLICT_SCORE_THRESHOLD:
            reward += 0.15
        else:
            reward -= 0.05

    elif action == "RejectAction":
        reward += R_CORRECT_MODE_SWITCH if conflict_score > CONFLICT_SCORE_THRESHOLD else -0.02

    elif action == "ApproveAction":
        reward += 0.03 if conflict_score == 0 else -0.02

    elif action == "BroadcastAlert":
        reward += 0.04

    elif action == "DeclareZoneClear":
        sev = severities.get(zone_id, 5.0)
        reward += 0.20 if sev <= 2.0 else -0.05

    elif action == "EscalateToExternal":
        avg_sev = sum(severities.values()) / len(severities) if severities else 5.0
        reward += 0.10 if avg_sev > 7.0 else -0.03

    elif action == "NoOp":
        reward -= 0.05

    return round(reward, 4)


# ---------------------------------------------------------------------------
# GRPOTrainer reward functions — CORRECT TRL signature
# ---------------------------------------------------------------------------

def compute_grpo_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Primary reward function for GRPOTrainer.

    TRL calls this as: reward_func(completions, prompts=prompts, ...)
    prompts arrive via **kwargs, NOT as a positional argument.

    Usage in grpo_train.ipynb:
        trainer = GRPOTrainer(
            ...
            reward_funcs=[compute_grpo_reward],
        )
    """
    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []
    for prompt, completion in zip(prompts, completions):
        action, params = _parse_action_from_completion(completion)
        state = _parse_state_from_prompt(prompt)
        rewards.append(_heuristic_reward(action, params, state))
    return rewards


def reward_action_validity(completions: list[str], **kwargs) -> list[float]:
    """
    Reward head: +0.20 if completion contains a valid action name, else -0.20.
    Trains the model to output recognisable action names.
    """
    valid_actions = {
        "DispatchAmbulance", "SetupFieldHospital", "RequestBloodSupply",
        "TriageZone", "BroadcastMedUpdate", "ClearRoad", "RouteTruck",
        "RefuelVehicle", "RequestAirDrop", "BroadcastLogistUpdate",
        "ApproveAction", "RejectAction", "OverrideAgent",
        "EscalateToExternal", "BroadcastAlert", "DeclareZoneClear",
    }
    rewards = []
    for completion in completions:
        found = any(a.lower() in completion.lower() for a in valid_actions)
        rewards.append(0.20 if found else -0.20)
    return rewards


def reward_format_compliance(completions: list[str], **kwargs) -> list[float]:
    """
    Reward head: checks for structured 'Action: X / Parameters: {...}' format.
    +0.20 = action + params + reasoning, +0.08 = action + params,
    +0.00 = action only, -0.20 = no action field.
    """
    rewards = []
    for completion in completions:
        has_action = bool(re.search(r"[Aa]ction\s*[:\-]", completion))
        has_params = bool(re.search(r"[Pp]arameters?\s*[:\-]", completion))
        has_reasoning = bool(re.search(r"[Rr]easoning\s*[:\-]", completion))
        if has_action and has_params and has_reasoning:
            rewards.append(0.20)
        elif has_action and has_params:
            rewards.append(0.08)
        elif has_action:
            rewards.append(0.00)
        else:
            rewards.append(-0.20)
    return rewards


def reward_zone_priority(completions: list[str], **kwargs) -> list[float]:
    """
    Reward head: +0.10 if agent targets the highest-severity zone.
    Encourages triage-aware prioritisation over random zone selection.
    """
    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []
    for prompt, completion in zip(prompts, completions):
        state = _parse_state_from_prompt(prompt)
        severities = state.get("zone_severities", {})
        if not severities:
            rewards.append(0.0)
            continue
        highest_zone = max(severities, key=lambda z: severities[z])
        _, params = _parse_action_from_completion(completion)
        rewards.append(0.10 if params.get("zone_id", -1) == highest_zone else -0.03)
    return rewards