# app/agents.py
# Agent definitions for OmegaResponse.
# Each agent has:
#   - A role-specific system prompt
#   - A prompt builder that formats env observation into an LLM-ready string
#   - A response parser that extracts action + params from LLM output
#
# Groq client is optional — used only in episode_runner.py, not during GRPO training.

from __future__ import annotations
import os
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.environment import AgentRole

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MED_AGENT_SYSTEM_PROMPT = """You are MedAgent, the medical coordinator for a disaster response operation.
You represent the WASH Cluster (WHO) in a UN humanitarian response.

YOUR PARTIAL OBSERVATION INCLUDES:
- Hospital capacity (fraction 0-1)
- Casualties per zone (number of injured people)
- Zone severity (0-10 scale, 10 = catastrophic)
- Medical supply levels per zone
- Recent broadcast messages from other agents

YOUR AVAILABLE ACTIONS (choose exactly one per turn):
1. DispatchAmbulance     — Send ambulance to a zone. params: {"zone_id": <0-4>}
2. SetupFieldHospital    — Establish field hospital in a zone. params: {"zone_id": <0-4>}
3. RequestBloodSupply    — Request blood/medical supplies. params: {"zone_id": <0-4>}
4. TriageZone            — Triage and treat casualties in place. params: {"zone_id": <0-4>}
5. BroadcastMedUpdate    — Share your medical assessment. params: {"message": "<text>", "belief": {"priority_zone": <0-4>, "hospital_capacity": <float>}}

DECISION RULES:
- Always prioritise zones with highest severity AND casualties
- Do NOT dispatch ambulance to a zone if LogistAgent already routed a truck there this step
- Set up field hospitals when hospital_capacity < 0.5
- Broadcast your priority zone every 5 steps to coordinate with LogistAgent
- If you see a broadcast from LogistAgent clearing a road, dispatch ambulance to that zone immediately

OUTPUT FORMAT (follow exactly):
Action: <ActionName>
Parameters: {"zone_id": <int>}
Reasoning: <one sentence explaining your choice>"""


LOGIST_AGENT_SYSTEM_PROMPT = """You are LogistAgent, the logistics coordinator for a disaster response operation.
You represent the Logistics Cluster (WFP) in a UN humanitarian response.

YOUR PARTIAL OBSERVATION INCLUDES:
- Road conditions per zone (clear / blocked / damaged)
- Vehicle locations (vehicle_id -> zone_id mapping)
- Fuel levels per vehicle (fraction 0-1)
- Global supply stock level
- Recent broadcast messages from other agents

YOUR AVAILABLE ACTIONS (choose exactly one per turn):
1. ClearRoad             — Clear a blocked/damaged road. params: {"zone_id": <0-4>}
2. RouteTruck            — Route a supply truck to a zone. params: {"zone_id": <0-4>, "vehicle_id": <0-3>}
3. RefuelVehicle         — Refuel a vehicle. params: {"vehicle_id": <0-3>}
4. RequestAirDrop        — Request aerial supply drop. params: {"zone_id": <0-4>}
5. BroadcastLogistUpdate — Share road/supply status. params: {"message": "<text>", "belief": {"cleared_zone": <0-4>, "supply_stock": <float>}}

DECISION RULES:
- FIRST clear roads to high-severity zones before routing trucks
- Do NOT route a truck to a zone where MedAgent already dispatched an ambulance this step
- Refuel vehicles when fuel < 0.3
- Broadcast when you clear a road so MedAgent can dispatch ambulance immediately
- Use RequestAirDrop for zones with supply_level < 0.2 and blocked roads

OUTPUT FORMAT (follow exactly):
Action: <ActionName>
Parameters: {"zone_id": <int>}
Reasoning: <one sentence explaining your choice>"""


COMMAND_AGENT_SYSTEM_PROMPT = """You are CommandAgent, the OCHA Humanitarian Coordinator overseeing the disaster response.
You have FULL VISIBILITY into all zones, all agent actions, and all broadcasts.

YOUR TWO OPERATING MODES:
- MonitorMode (default): Observe, score agent actions, detect conflicts, log anomalies.
  Stay in MonitorMode unless conflict_score > 2.
- InterventionMode (auto-triggered when conflict_score > 2): Actively intervene.
  Override agents, reject conflicting actions, escalate if situation is critical.

YOUR AVAILABLE ACTIONS (choose exactly one per turn):
1. ApproveAction      — Approve a pending agent action. params: {"target_action": "<action_name>"}
2. RejectAction       — Reject a conflicting action. params: {"target_action": "<action_name>"}
3. OverrideAgent      — Override an agent. params: {"target_agent": "<MedAgent|LogistAgent>", "override_action": "<action>"}
4. EscalateToExternal — Escalate to external agencies (avg severity > 7). params: {"reason": "<text>"}
5. BroadcastAlert     — Broadcast coordination alert to all agents. params: {"message": "<text>", "belief": {}}
6. DeclareZoneClear   — Officially declare a zone recovered. params: {"zone_id": <0-4>}

DECISION RULES IN MonitorMode:
- If two agents acted on the same zone this step → conflict detected
- BroadcastAlert to coordinate agents proactively
- DeclareZoneClear when zone severity <= 2.0
- ApproveAction when agents are coordinating correctly

DECISION RULES IN InterventionMode (conflict_score > 2):
- RejectAction for the lower-priority conflicting action
- OverrideAgent to redirect the misallocated agent
- EscalateToExternal if average zone severity > 7.0
- Switch back to MonitorMode once conflict_score returns to 0

OUTPUT FORMAT (follow exactly):
Mode: <MonitorMode|InterventionMode>
Action: <ActionName>
Parameters: {"zone_id": <int>}
Reasoning: <one sentence explaining your choice>"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_med_prompt(obs: dict) -> str:
    zone_lines = []
    for z in obs.get("zones", []):
        zone_lines.append(
            f"  Zone {z['zone_id']}: severity={z['severity']} | "
            f"casualties={z['casualties']} | "
            f"hospital_nearby={z['hospital_nearby']} | "
            f"supply_level={z['supply_level']}"
        )

    broadcasts = obs.get("broadcast_log", [])
    broadcast_lines = [
        f"  [{b.get('from', '?')} @ step {b.get('step', '?')}]: {b.get('message', '')}"
        for b in broadcasts[-3:]
    ] or ["  (no recent broadcasts)"]

    return f"""You are the MedAgent.
Step: {obs.get('step', 0)} / 300
hospital_capacity={obs.get('hospital_capacity', '?')}

ZONE STATUS:
{chr(10).join(zone_lines)}

RECENT BROADCASTS:
{chr(10).join(broadcast_lines)}

Choose your action. Prioritise the zone with the highest severity and casualties.
Output format:
Action: <ActionName>
Parameters: {{"zone_id": <int>}}
Reasoning: <one sentence>"""


def build_logist_prompt(obs: dict) -> str:
    road_lines = [
        f"  Zone {zid} road: {status}"
        for zid, status in obs.get("road_conditions", {}).items()
    ]
    vehicle_lines = [
        f"  Vehicle {vid} -> Zone {zid}"
        for vid, zid in obs.get("vehicle_locations", {}).items()
    ]
    fuel_lines = [
        f"  Vehicle {vid}: {fuel:.0%} fuel"
        for vid, fuel in obs.get("fuel_levels", {}).items()
    ]
    broadcasts = obs.get("broadcast_log", [])
    broadcast_lines = [
        f"  [{b.get('from', '?')} @ step {b.get('step', '?')}]: {b.get('message', '')}"
        for b in broadcasts[-3:]
    ] or ["  (no recent broadcasts)"]

    return f"""You are the LogistAgent.
Step: {obs.get('step', 0)} / 300
supply_stock={obs.get('supply_stock', '?')}

ROAD CONDITIONS:
{chr(10).join(road_lines) or '  (no data)'}

VEHICLE LOCATIONS:
{chr(10).join(vehicle_lines) or '  (no vehicles)'}

FUEL LEVELS:
{chr(10).join(fuel_lines) or '  (no data)'}

RECENT BROADCASTS:
{chr(10).join(broadcast_lines)}

Choose your action. Clear roads to high-severity zones before routing trucks.
Output format:
Action: <ActionName>
Parameters: {{"zone_id": <int>}}
Reasoning: <one sentence>"""


def build_command_prompt(obs: dict) -> str:
    zone_lines = []
    for z in obs.get("zones", []):
        zone_lines.append(
            f"  Zone {z['zone_id']}: severity={z['severity']} | "
            f"casualties={z.get('casualties', '?')} | "
            f"road={z.get('road_status', '?')} | "
            f"supply={z.get('supply_level', '?')}"
        )

    recent_actions = obs.get("action_history", [])[-6:]
    action_lines = [
        f"  step={a['step']} {a['agent']}: {a['action']} {a.get('params', {})}"
        for a in recent_actions
    ] or ["  (no recent actions)"]

    anomalies = obs.get("anomaly_log", []) or ["  (no anomalies)"]
    broadcasts = obs.get("broadcast_log", [])
    broadcast_lines = [
        f"  [{b.get('from', '?')} @ step {b.get('step', '?')}]: {b.get('message', '')}"
        for b in broadcasts[-3:]
    ] or ["  (no recent broadcasts)"]

    return f"""You are the CommandAgent.
Step: {obs.get('step', 0)} / 300
command_mode={obs.get('command_mode', 'MonitorMode')}
conflict_score={obs.get('conflict_score', 0)}
hospital_capacity={obs.get('hospital_capacity', '?')}

ZONE STATUS (full visibility):
{chr(10).join(zone_lines)}

RECENT AGENT ACTIONS:
{chr(10).join(action_lines)}

ANOMALY LOG:
{chr(10).join(str(a) for a in anomalies[-3:])}

RECENT BROADCASTS:
{chr(10).join(broadcast_lines)}

If conflict_score > 2, switch to InterventionMode and intervene.
Output format:
Mode: <MonitorMode|InterventionMode>
Action: <ActionName>
Parameters: {{"zone_id": <int>}}
Reasoning: <one sentence>"""


# ---------------------------------------------------------------------------
# Response parser — FIXED: single-escaped regex, no double backslashes
# ---------------------------------------------------------------------------

def parse_agent_response(response_text: str) -> dict:
    """
    Parse LLM output into a structured action dict.
    Handles both well-formatted and messy outputs.
    """
    result = {
        "action": "BroadcastAlert",
        "params": {"zone_id": 0},
        "reasoning": "",
        "mode": "MonitorMode",
        "raw": response_text,
    }

    # Mode (CommandAgent)
    mode_match = re.search(r"[Mm]ode\s*[:\-]\s*(\w+Mode)", response_text)
    if mode_match:
        result["mode"] = mode_match.group(1).strip()

    # Action name — structured format
    action_match = re.search(
        r"[Aa]ction\s*[:\-]\s*([A-Za-z][A-Za-z0-9]+)",
        response_text
    )
    if action_match:
        result["action"] = action_match.group(1).strip()
    else:
        # Keyword scan fallback
        known_actions = [
            "DispatchAmbulance", "SetupFieldHospital", "RequestBloodSupply",
            "TriageZone", "BroadcastMedUpdate",
            "ClearRoad", "RouteTruck", "RefuelVehicle",
            "RequestAirDrop", "BroadcastLogistUpdate",
            "ApproveAction", "RejectAction", "OverrideAgent",
            "EscalateToExternal", "BroadcastAlert", "DeclareZoneClear",
        ]
        for ka in known_actions:
            if ka.lower() in response_text.lower():
                result["action"] = ka
                break

    # Parameters — try JSON block first
    params_match = re.search(r"[Pp]arameters?\s*[:\-]\s*(\{[^}]+\})", response_text)
    if params_match:
        try:
            result["params"] = json.loads(params_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: extract zone_id inline
    if "zone_id" not in result["params"]:
        zone_match = re.search(r"zone_id[\"']?\s*[:\-]\s*(\d)", response_text)
        if zone_match:
            result["params"]["zone_id"] = int(zone_match.group(1))

    # Fallback: extract vehicle_id inline
    if "vehicle_id" not in result["params"]:
        veh_match = re.search(r"vehicle_id[\"']?\s*[:\-]\s*(\d)", response_text)
        if veh_match:
            result["params"]["vehicle_id"] = int(veh_match.group(1))

    # target_agent (CommandAgent overrides)
    if "target_agent" not in result["params"]:
        ta_match = re.search(
            r"target_agent[\"']?\s*[:\-]\s*[\"']?([A-Za-z]+Agent)[\"']?",
            response_text
        )
        if ta_match:
            result["params"]["target_agent"] = ta_match.group(1)

    # Reasoning
    reasoning_match = re.search(r"[Rr]easoning\s*[:\-]\s*(.+)", response_text)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Groq client helper (episode_runner.py only — never called during GRPO)
# ---------------------------------------------------------------------------

def get_groq_client():
    try:
        import groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Export it before running episode_runner.py")
    # Set a client-level timeout so the first request cannot hang indefinitely.
    return groq.Groq(api_key=api_key, timeout=30.0)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Call Groq API with timeout and retry protection."""
    import os
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")

    # timeout=20 total, connect=5 — fails fast instead of hanging 60s
    client = Groq(
        api_key=api_key,
        timeout=20.0,
        max_retries=1,      # default is 2 — reduces hang time on failure
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Agent config registry
# ---------------------------------------------------------------------------

AGENT_CONFIG = {
    "MedAgent": {
        "system_prompt": MED_AGENT_SYSTEM_PROMPT,
        "build_prompt":  build_med_prompt,
    },
    "LogistAgent": {
        "system_prompt": LOGIST_AGENT_SYSTEM_PROMPT,
        "build_prompt":  build_logist_prompt,
    },
    "CommandAgent": {
        "system_prompt": COMMAND_AGENT_SYSTEM_PROMPT,
        "build_prompt":  build_command_prompt,
    },
}


def _highest_severity_zone(obs: dict) -> int:
    """
    Return the zone_id with the highest severity from obs.
    Works for MedAgent obs (has 'zones' list) and CommandAgent obs (same).
    Falls back to 0 if no zone data available.
    """
    zones = obs.get("zones", [])
    if not zones:
        return 0
    return max(zones, key=lambda z: z.get("severity", 0))["zone_id"]


def get_agent_action(agent_role: str, obs: dict, use_groq: bool = False) -> dict:
    """
    Get an action for the given agent role.

    Args:
        agent_role: "MedAgent" | "LogistAgent" | "CommandAgent"
        obs:        partial observation dict from env.get_observation()
        use_groq:   True  = call Groq API (local testing only)
                    False = return smart rule-based default (used by rule_based_agent.py)

    Returns:
        Parsed action dict: {"action": str, "params": dict, "reasoning": str, ...}
    """
    config = AGENT_CONFIG.get(agent_role)
    if config is None:
        raise ValueError(f"Unknown agent role: {agent_role}")

    user_prompt = config["build_prompt"](obs)

    if use_groq:
        raw = call_llm(config["system_prompt"], user_prompt)
        return parse_agent_response(raw)

    # Smart rule-based default — picks highest severity zone, not hardcoded 0
    priority_zone = _highest_severity_zone(obs)

    default_actions = {
        "MedAgent":     ("DispatchAmbulance",     {"zone_id": priority_zone}),
        "LogistAgent":  ("ClearRoad",             {"zone_id": priority_zone}),
        "CommandAgent": ("BroadcastAlert",        {"message": "Monitoring all zones", "belief": {}}),
    }
    action, params = default_actions[agent_role]

    return {
        "action":    action,
        "params":    params,
        "reasoning": f"Rule-based: targeting highest severity zone {priority_zone}",
        "mode":      "MonitorMode",
        "raw":       "",
    }