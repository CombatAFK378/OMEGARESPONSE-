# rule_based_agent.py
# Two responsibilities:
#   1. Rule-based if/else agent that plays a full episode against the env
#      (used as the BEFORE-training baseline — expected score ~0.3)
#   2. Generates prompts_dataset.jsonl with 200 randomised disaster scenario
#      prompts formatted exactly as GRPOTrainer expects (prompts only, no actions)
#
# Run modes:
#   python rule_based_agent.py --mode episode   → play one episode, print score
#   python rule_based_agent.py --mode generate  → write prompts_dataset.jsonl
#   python rule_based_agent.py --mode both      → episode first, then generate

import argparse
import json
import random
import sys
from pathlib import Path

# environment.py and agents.py must exist at app/
from app.environment import DisasterEnvironment, AgentRole, RoadStatus, CommandMode
from app.agents import build_med_prompt, build_logist_prompt, build_command_prompt

NUM_ZONES = 5
DATASET_SIZE = 2000
OUTPUT_PATH = Path("prompts_dataset.jsonl")


def normalize_score(total_reward: float, total_actions: int) -> float:
    """
    Normalize cumulative reward to a per-action score for demo comparison.
    Baseline rule-based ~0.175, post-GRPO target ~0.5-0.8
    """
    return round(total_reward / max(total_actions, 1), 4)


# ---------------------------------------------------------------------------
# Rule-based decision logic — one function per agent
# ---------------------------------------------------------------------------

def rule_based_med(obs: dict) -> tuple[str, dict]:
    """
    MedAgent if/else rules:
    1. If hospital_capacity < 0.5 → SetupFieldHospital at highest-severity zone
    2. If any zone has casualties > 0 and road is clear (from broadcasts) → DispatchAmbulance
    3. If step % 5 == 0 → BroadcastMedUpdate
    4. Else → TriageZone at highest-severity zone
    """
    step = obs.get("step", 0)
    hospital_capacity = obs.get("hospital_capacity", 1.0)
    zones = obs.get("zones", [])
    broadcasts = obs.get("broadcast_log", [])

    if not zones:
        return "BroadcastMedUpdate", {"message": "No zone data", "belief": {}}

    # Sort zones by severity descending
    sorted_zones = sorted(zones, key=lambda z: z.get("severity", 0), reverse=True)
    top_zone = sorted_zones[0]["zone_id"]

    # Check if any zone had its road cleared recently (via broadcast)
    cleared_zones = set()
    for b in broadcasts:
        msg = b.get("message", "").lower()
        belief = b.get("belief", {})
        if "cleared" in msg or "clear" in msg:
            if "cleared_zone" in belief:
                cleared_zones.add(int(belief["cleared_zone"]))

    # Rule 1: low hospital capacity
    if hospital_capacity < 0.5:
        return "SetupFieldHospital", {"zone_id": top_zone}

    # Rule 2: dispatch ambulance to cleared road zone with casualties
    for z in sorted_zones:
        if z["zone_id"] in cleared_zones and z.get("casualties", 0) > 0:
            return "DispatchAmbulance", {"zone_id": z["zone_id"]}

    # Rule 3: dispatch to highest severity zone with casualties
    for z in sorted_zones:
        if z.get("casualties", 0) > 0:
            return "DispatchAmbulance", {"zone_id": z["zone_id"]}

    # Rule 4: broadcast every 5 steps
    if step % 5 == 0:
        return "BroadcastMedUpdate", {
            "message": f"Priority zone {top_zone} severity={sorted_zones[0].get('severity', 0):.1f}",
            "belief": {"priority_zone": top_zone, "hospital_capacity": hospital_capacity},
        }

    # Rule 5: triage
    return "TriageZone", {"zone_id": top_zone}


def rule_based_logist(obs: dict) -> tuple[str, dict]:
    """
    LogistAgent if/else rules:
    1. If any vehicle has fuel < 0.3 → RefuelVehicle
    2. If any zone has blocked/damaged road → ClearRoad (highest severity blocked zone)
    3. If any zone has supply_level missing (inferred from broadcasts) → RouteTruck
    4. If step % 5 == 0 → BroadcastLogistUpdate
    5. Else → RequestAirDrop at zone with worst road
    """
    step = obs.get("step", 0)
    road_conditions = obs.get("road_conditions", {})
    vehicle_locations = obs.get("vehicle_locations", {})
    fuel_levels = obs.get("fuel_levels", {})
    supply_stock = obs.get("supply_stock", 1.0)
    broadcasts = obs.get("broadcast_log", [])

    # Rule 1: refuel low vehicles
    for vid, fuel in fuel_levels.items():
        if fuel < 0.3:
            return "RefuelVehicle", {"vehicle_id": int(vid)}

    # Find highest-severity zone from broadcasts (MedAgent broadcasts priority zone)
    priority_zone = 0
    for b in reversed(broadcasts):
        belief = b.get("belief", {})
        if "priority_zone" in belief:
            priority_zone = int(belief["priority_zone"])
            break

    # Rule 2: clear blocked/damaged roads — prioritise priority_zone first
    blocked = {
        int(zid): status
        for zid, status in road_conditions.items()
        if status in ("blocked", "damaged")
    }
    if blocked:
        # Prefer priority zone if blocked, else pick any blocked zone
        if priority_zone in blocked:
            return "ClearRoad", {"zone_id": priority_zone}
        return "ClearRoad", {"zone_id": next(iter(blocked))}

    # Rule 3: route truck to priority zone if not already there
    vehicles_at_priority = [
        vid for vid, zid in vehicle_locations.items()
        if int(zid) == priority_zone
    ]
    if not vehicles_at_priority:
        # Find vehicle with most fuel
        best_vehicle = max(fuel_levels, key=lambda v: fuel_levels[v], default=0)
        return "RouteTruck", {"zone_id": priority_zone, "vehicle_id": int(best_vehicle)}

    # Rule 4: broadcast road status every 5 steps
    if step % 5 == 0:
        clear_zones = [
            int(zid) for zid, status in road_conditions.items()
            if status == "clear"
        ]
        cleared_zone = clear_zones[0] if clear_zones else priority_zone
        return "BroadcastLogistUpdate", {
            "message": f"Road cleared for Zone {cleared_zone}",
            "belief": {"cleared_zone": cleared_zone, "supply_stock": supply_stock},
        }

    # Rule 5: airdrop to priority zone
    return "RequestAirDrop", {"zone_id": priority_zone}


def rule_based_command(obs: dict) -> tuple[str, dict]:
    """
    CommandAgent if/else rules:
    MonitorMode:
      1. If any zone severity <= 2 → DeclareZoneClear
      2. Broadcast coordination alert every 5 steps
      3. If conflict_score > threshold → ApproveAction (passive)
    InterventionMode (conflict_score > 2):
      1. RejectAction for conflicting agent
      2. If avg severity > 7 → EscalateToExternal
      3. OverrideAgent to redirect
    """
    step = obs.get("step", 0)
    command_mode = obs.get("command_mode", "MonitorMode")
    conflict_score = obs.get("conflict_score", 0)
    zones = obs.get("zones", [])
    action_history = obs.get("action_history", [])

    if not zones:
        return "BroadcastAlert", {"message": "Awaiting zone data", "belief": {}}

    avg_severity = sum(z.get("severity", 0) for z in zones) / len(zones)

    # Find zones that are clear (severity <= 2)
    clear_zones = [z["zone_id"] for z in zones if z.get("severity", 10) <= 2.0]

    if command_mode == "MonitorMode":
        # Rule 1: declare clear zones
        if clear_zones:
            return "DeclareZoneClear", {"zone_id": clear_zones[0]}

        # Rule 2: broadcast every 5 steps
        if step % 5 == 0:
            return "BroadcastAlert", {
                "message": f"CommandAgent monitoring: avg_severity={avg_severity:.1f}, conflict_score={conflict_score}",
                "belief": {"avg_severity": avg_severity},
            }

        # Rule 3: approve recent action
        if action_history:
            last_action = action_history[-1]["action"]
            return "ApproveAction", {"target_action": last_action}

        return "BroadcastAlert", {"message": "All systems monitored", "belief": {}}

    else:
        # InterventionMode
        # Rule 1: escalate if critical
        if avg_severity > 7.0:
            return "EscalateToExternal", {"reason": f"Critical: avg_severity={avg_severity:.1f}"}

        # Rule 2: reject last conflicting action
        if action_history:
            last_action = action_history[-1]["action"]
            return "RejectAction", {"target_action": last_action}

        # Rule 3: override agent
        return "OverrideAgent", {
            "target_agent": "MedAgent",
            "override_action": "TriageZone",
        }


# ---------------------------------------------------------------------------
# Episode runner using rule-based agent
# ---------------------------------------------------------------------------

def run_rule_based_episode(seed: int = 42, verbose: bool = True) -> float:
    """
    Run one full episode (300 steps) using the rule-based agent.
    Returns normalized per-action score.
    """
    env = DisasterEnvironment(seed=seed)
    state = env.reset(seed=seed)

    total_reward = 0.0
    step_rewards = []

    rotation = [AgentRole.MED, AgentRole.LOGIST, AgentRole.COMMAND]
    rule_fns = {
        AgentRole.MED:     rule_based_med,
        AgentRole.LOGIST:  rule_based_logist,
        AgentRole.COMMAND: rule_based_command,
    }

    done = False
    sub_turn = 0  # tracks position in Med→Logist→Command rotation

    while not done:
        agent = rotation[sub_turn % 3]
        obs = env.get_observation(agent)
        action, params = rule_fns[agent](obs)

        result = env.step(agent, action, params)
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward
        step_rewards.append(reward)

        if verbose and env.state.step % 50 == 0 and sub_turn % 3 == 0:
            info = result.get("info", {})
            print(
                f"  Step {env.state.step:3d}/300 | "
                f"reward={reward:+.4f} | "
                f"cumulative={total_reward:.3f} | "
                f"mode={info.get('command_mode', '?')} | "
                f"conflicts={info.get('conflict_score', '?')}"
            )

        sub_turn += 1

    if verbose:
        normalized = normalize_score(total_reward, env.state.total_actions)
        print(f"\n{'='*50}")
        print(f"  EPISODE COMPLETE")
        print(f"  Raw cumulative reward : {total_reward:.4f}  (300 steps x 3 agents)")
        print(f"  Normalized per-action : {normalized:.4f}  <- use this for demo")
        print(f"  Final state           : step={env.state.step}, done={env.state.done}")
        print(f"  Conflicts             : {env.state.total_conflicts}/{env.state.total_actions} actions")
        zones_recovered = sum(1 for z in env.state.zones if z.severity == 0.0)
        print(f"  Zones recovered       : {zones_recovered}/5")
        print(f"{'='*50}\n")

    return normalize_score(total_reward, env.state.total_actions)


# ---------------------------------------------------------------------------
# Prompt dataset generator
# ---------------------------------------------------------------------------

def _random_zone_block(rng: random.Random) -> dict:
    """Generate a single randomised zone state dict."""
    return {
        "zone_id": None,  # filled by caller
        "severity": round(rng.uniform(0.5, 9.8), 2),
        "casualties": rng.randint(0, 60),
        "road_status": rng.choice(["clear", "blocked", "damaged"]),
        "hospital_nearby": rng.random() > 0.5,
        "supply_level": round(rng.uniform(0.05, 0.95), 2),
    }


def _format_med_prompt(rng: random.Random, step: int) -> str:
    """Build a MedAgent prompt string from a random scenario."""
    zones = []
    for i in range(NUM_ZONES):
        z = _random_zone_block(rng)
        z["zone_id"] = i
        zones.append(z)

    hospital_capacity = round(rng.uniform(0.2, 1.0), 2)

    # Simulate 0-2 recent broadcasts
    broadcasts = []
    if rng.random() > 0.5:
        cleared_zone = rng.randint(0, NUM_ZONES - 1)
        broadcasts.append(
            f"  [LogistAgent @ step {max(0, step-2)}]: Road cleared for Zone {cleared_zone} | "
            f"belief: cleared_zone={cleared_zone}"
        )
    if rng.random() > 0.7:
        broadcasts.append(
            f"  [CommandAgent @ step {max(0, step-1)}]: Monitoring — conflict_score={rng.randint(0,3)}"
        )

    zone_lines = "\n".join(
        f"  Zone {z['zone_id']}: severity={z['severity']} | "
        f"casualties={z['casualties']} | "
        f"hospital_nearby={z['hospital_nearby']} | "
        f"supply_level={z['supply_level']}"
        for z in zones
    )
    broadcast_block = "\n".join(broadcasts) if broadcasts else "  (no recent broadcasts)"

    return (
        f"You are the MedAgent.\n"
        f"Step: {step} / 300\n"
        f"hospital_capacity={hospital_capacity}\n\n"
        f"ZONE STATUS:\n{zone_lines}\n\n"
        f"RECENT BROADCASTS:\n{broadcast_block}\n\n"
        f"Choose your action. Prioritise the zone with the highest severity and casualties.\n"
        f"Your available actions: DispatchAmbulance, SetupFieldHospital, RequestBloodSupply, TriageZone, BroadcastMedUpdate\n\n"
        f"Output format:\n"
        f"Action: <ActionName>\n"
        f"Parameters: {{\"zone_id\": <int>}}\n"
        f"Reasoning: <one sentence>"
    )


def _format_logist_prompt(rng: random.Random, step: int) -> str:
    """Build a LogistAgent prompt string from a random scenario."""
    road_conditions = {
        i: rng.choice(["clear", "blocked", "damaged"])
        for i in range(NUM_ZONES)
    }
    vehicle_locations = {v: rng.randint(0, NUM_ZONES - 1) for v in range(4)}
    fuel_levels = {v: round(rng.uniform(0.1, 1.0), 2) for v in range(4)}
    supply_stock = round(rng.uniform(0.2, 1.0), 2)

    road_lines = "\n".join(
        f"  Zone {zid} road: {status}"
        for zid, status in road_conditions.items()
    )
    vehicle_lines = "\n".join(
        f"  Vehicle {vid} -> Zone {zid}"
        for vid, zid in vehicle_locations.items()
    )
    fuel_lines = "\n".join(
        f"  Vehicle {vid}: {fuel:.0%} fuel"
        for vid, fuel in fuel_levels.items()
    )

    priority_zone = rng.randint(0, NUM_ZONES - 1)
    broadcasts = []
    if rng.random() > 0.4:
        broadcasts.append(
            f"  [MedAgent @ step {max(0, step-1)}]: Priority zone {priority_zone} "
            f"severity={round(rng.uniform(5,9.5),1)} | belief: priority_zone={priority_zone}"
        )

    broadcast_block = "\n".join(broadcasts) if broadcasts else "  (no recent broadcasts)"

    return (
        f"You are the LogistAgent.\n"
        f"Step: {step} / 300\n"
        f"supply_stock={supply_stock}\n\n"
        f"ROAD CONDITIONS:\n{road_lines}\n\n"
        f"VEHICLE LOCATIONS:\n{vehicle_lines}\n\n"
        f"FUEL LEVELS:\n{fuel_lines}\n\n"
        f"RECENT BROADCASTS:\n{broadcast_block}\n\n"
        f"Choose your action. Clear roads to high-severity zones before routing trucks.\n"
        f"Your available actions: ClearRoad, RouteTruck, RefuelVehicle, RequestAirDrop, BroadcastLogistUpdate\n\n"
        f"Output format:\n"
        f"Action: <ActionName>\n"
        f"Parameters: {{\"zone_id\": <int>}}\n"
        f"Reasoning: <one sentence>"
    )


def _format_command_prompt(rng: random.Random, step: int) -> str:
    """Build a CommandAgent prompt string from a random scenario."""
    zones = []
    for i in range(NUM_ZONES):
        z = _random_zone_block(rng)
        z["zone_id"] = i
        zones.append(z)

    conflict_score = rng.randint(0, 5)
    command_mode = "InterventionMode" if conflict_score > 2 else "MonitorMode"
    hospital_capacity = round(rng.uniform(0.2, 1.0), 2)
    avg_severity = round(sum(z["severity"] for z in zones) / NUM_ZONES, 2)

    zone_lines = "\n".join(
        f"  Zone {z['zone_id']}: severity={z['severity']} | "
        f"casualties={z['casualties']} | "
        f"road={z['road_status']} | "
        f"supply={z['supply_level']}"
        for z in zones
    )

    # Simulate recent agent actions
    actions_pool = [
        ("MedAgent",    "DispatchAmbulance",    {"zone_id": rng.randint(0, 4)}),
        ("LogistAgent", "RouteTruck",           {"zone_id": rng.randint(0, 4), "vehicle_id": rng.randint(0, 3)}),
        ("MedAgent",    "TriageZone",           {"zone_id": rng.randint(0, 4)}),
        ("LogistAgent", "ClearRoad",            {"zone_id": rng.randint(0, 4)}),
    ]
    recent_actions = rng.sample(actions_pool, k=min(3, len(actions_pool)))
    action_lines = "\n".join(
        f"  step={step - rng.randint(1,3)} {ag}: {act} {params}"
        for ag, act, params in recent_actions
    )

    anomaly = ""
    if conflict_score > 2:
        anomaly = f"  step={step-1}: conflict_score={conflict_score} — duplicate dispatch detected"

    return (
        f"You are the CommandAgent.\n"
        f"Step: {step} / 300\n"
        f"command_mode={command_mode}\n"
        f"conflict_score={conflict_score}\n"
        f"hospital_capacity={hospital_capacity}\n\n"
        f"ZONE STATUS (full visibility):\n{zone_lines}\n\n"
        f"RECENT AGENT ACTIONS:\n{action_lines}\n\n"
        f"ANOMALY LOG:\n{anomaly if anomaly else '  (no anomalies)'}\n\n"
        f"RECENT BROADCASTS:\n  (no recent broadcasts)\n\n"
        f"If conflict_score > 2, switch to InterventionMode and intervene.\n"
        f"Your available actions: ApproveAction, RejectAction, OverrideAgent, EscalateToExternal, BroadcastAlert, DeclareZoneClear\n\n"
        f"Output format:\n"
        f"Mode: <MonitorMode|InterventionMode>\n"
        f"Action: <ActionName>\n"
        f"Parameters: {{\"zone_id\": <int>}}\n"
        f"Reasoning: <one sentence>"
    )


def generate_prompts_dataset(
    output_path: Path = OUTPUT_PATH,
    size: int = DATASET_SIZE,
    seed: int = 0,
) -> None:
    """
    Generate 200 randomised disaster scenario prompts and save to JSONL.

    Format per line (GRPOTrainer expects):
        {"prompt": "<state prompt string>"}

    Distribution: ~67 MedAgent, ~67 LogistAgent, ~66 CommandAgent prompts
    covering a range of steps (0-300) and scenario difficulty levels.
    """
    rng = random.Random(seed)
    builders = [_format_med_prompt, _format_logist_prompt, _format_command_prompt]
    agent_names = ["MedAgent", "LogistAgent", "CommandAgent"]

    records = []
    for i in range(size):
        # Round-robin agent distribution
        builder = builders[i % 3]
        agent = agent_names[i % 3]

        # Spread steps across the full episode
        step = rng.randint(0, 295)

        prompt = builder(rng, step)
        records.append({
            "prompt": prompt,
            "agent": agent,   # metadata — GRPOTrainer ignores extra keys
            "scenario_id": i,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} prompts to {output_path}")
    print(f"  MedAgent:     {sum(1 for r in records if r['agent'] == 'MedAgent')}")
    print(f"  LogistAgent:  {sum(1 for r in records if r['agent'] == 'LogistAgent')}")
    print(f"  CommandAgent: {sum(1 for r in records if r['agent'] == 'CommandAgent')}")

    # Sanity-check: print first prompt
    print(f"\n--- Sample prompt [0] ---\n{records[0]['prompt'][:400]}...\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OmegaResponse rule-based agent")
    parser.add_argument(
        "--mode",
        choices=["episode", "generate", "both"],
        default="both",
        help=(
            "episode  = run one full rule-based episode and print score\n"
            "generate = write prompts_dataset.jsonl (200 prompts)\n"
            "both     = episode first, then generate (default)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output JSONL path")
    args = parser.parse_args()

    if args.mode in ("episode", "both"):
        print("\n=== Rule-Based Baseline Episode ===")
        score = run_rule_based_episode(seed=args.seed, verbose=True)
        print(f"Baseline normalized score: {score:.4f}  (target: ~0.175 | post-GRPO target: ~0.5-0.8)\n")

    if args.mode in ("generate", "both"):
        print("=== Generating prompts_dataset.jsonl ===")
        generate_prompts_dataset(
            output_path=Path(args.output),
            size=DATASET_SIZE,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()