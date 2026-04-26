# app/environment.py
# Core environment logic for OmegaResponse.
# Handles zone state, step rotation (Med -> Logist -> Command),
# partial observability, negotiation triggers, and long-horizon dependencies.

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# OpenEnv compliance: subclass the official base class
from openenv.core import Environment as OpenEnvBase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ZONES = 5
MAX_STEPS = 300
NEGOTIATION_INTERVAL = 5          # broadcast round every N steps
MILESTONE_INTERVAL = 20           # milestone reward check every N steps
CONFLICT_SCORE_THRESHOLD = 2      # CommandAgent switches to InterventionMode
LONG_HORIZON_DELAY_MIN = 40       # early action effect appears after this many steps
LONG_HORIZON_DELAY_MAX = 100


class AgentRole(str, Enum):
    MED = "MedAgent"
    LOGIST = "LogistAgent"
    COMMAND = "CommandAgent"


class RoadStatus(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    DAMAGED = "damaged"


class CommandMode(str, Enum):
    MONITOR = "MonitorMode"
    INTERVENTION = "InterventionMode"


# ---------------------------------------------------------------------------
# Zone dataclass
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    zone_id: int
    severity: float          # 0-10; 0 = fully recovered
    casualties: int
    road_status: RoadStatus
    hospital_nearby: bool
    supply_level: float      # 0-1 fraction
    # internal tracking
    idle_resources_steps: int = 0     # steps resources were idle nearby
    ambulance_en_route: bool = False
    road_cleared_recently: bool = False
    resources_available: bool = False  # True if supply/vehicle nearby

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "severity": round(self.severity, 2),
            "casualties": self.casualties,
            "road_status": self.road_status.value,
            "hospital_nearby": self.hospital_nearby,
            "supply_level": round(self.supply_level, 2),
            "ambulance_en_route": self.ambulance_en_route,
        }


# ---------------------------------------------------------------------------
# Global environment state
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentState:
    step: int = 0
    done: bool = False
    zones: List[Zone] = field(default_factory=list)

    # Agent-visible state (partial observability)
    hospital_capacity: float = 1.0     # fraction 0-1
    vehicle_locations: Dict[int, int] = field(default_factory=dict)  # vehicle_id -> zone_id
    fuel_levels: Dict[int, float] = field(default_factory=dict)       # vehicle_id -> 0-1
    supply_stock: float = 1.0          # global supply stock fraction

    # CommandAgent oversight state
    command_mode: CommandMode = CommandMode.MONITOR
    conflict_score: int = 0
    action_history: List[dict] = field(default_factory=list)
    interventions_total: int = 0
    interventions_correct: int = 0
    anomaly_log: List[str] = field(default_factory=list)

    # Negotiation / broadcast state
    broadcast_log: List[dict] = field(default_factory=list)
    coordinated_followups: int = 0

    # Conflict tracking
    total_actions: int = 0
    total_conflicts: int = 0

    # Long-horizon deferred effects queue: list of (trigger_step, effect_fn_name, kwargs)
    deferred_effects: List[dict] = field(default_factory=list)

    # Current agent turn in rotation
    current_agent_idx: int = 0  # 0=Med, 1=Logist, 2=Command
    agent_rotation: List[AgentRole] = field(default_factory=lambda: [
        AgentRole.MED, AgentRole.LOGIST, AgentRole.COMMAND
    ])

    @property
    def current_agent(self) -> AgentRole:
        return self.agent_rotation[self.current_agent_idx]

    def advance_agent(self):
        self.current_agent_idx = (self.current_agent_idx + 1) % 3
        if self.current_agent_idx == 0:
            self.step += 1  # full round = 1 step


# ---------------------------------------------------------------------------
# DisasterEnvironment
# ---------------------------------------------------------------------------

class DisasterEnvironment(OpenEnvBase):
    """
    OpenEnv-compliant disaster response environment.

    Lifecycle:
        env = DisasterEnvironment()
        state = env.reset()
        obs   = env.get_observation(AgentRole.MED)
        result = env.step(AgentRole.MED, "DispatchAmbulance", {"zone_id": 2})
        # result = {"obs": ..., "reward": ..., "done": ..., "info": ...}
    """

    VALID_ACTIONS = {
        AgentRole.MED: [
            "DispatchAmbulance",
            "SetupFieldHospital",
            "RequestBloodSupply",
            "TriageZone",
            "BroadcastMedUpdate",
        ],
        AgentRole.LOGIST: [
            "ClearRoad",
            "RouteTruck",
            "RefuelVehicle",
            "RequestAirDrop",
            "BroadcastLogistUpdate",
        ],
        AgentRole.COMMAND: [
            "ApproveAction",
            "RejectAction",
            "OverrideAgent",
            "EscalateToExternal",
            "BroadcastAlert",
            "DeclareZoneClear",
        ],
    }

    def __init__(self):
        self._rng = random.Random(None)
        self._env_state: EnvironmentState = self._build_initial_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, task_id: str = "omega_medium") -> dict:
        """Reset environment to a new randomised disaster scenario. Returns full state dict."""
        if seed is not None:
            self._rng = random.Random(seed)
        self._env_state = self._build_initial_state(task_id)
        return self._full_state_dict()

    def step(self, agent: AgentRole, action: str, params: dict) -> dict:
        """
        Execute one agent action.
        Returns {"obs": <agent-view>, "reward": float, "done": bool, "info": dict}
        """
        s = self._env_state
        if s.done:
            return {"obs": {}, "reward": 0.0, "done": True, "info": {"error": "episode_done"}}

        # Validate turn
        if agent != s.current_agent:
            return {
                "obs": self.get_observation(agent),
                "reward": -0.01,
                "done": False,
                "info": {"error": f"out_of_turn: expected {s.current_agent}, got {agent}"},
            }

        # Validate action
        if action not in self.VALID_ACTIONS.get(agent, []):
            return {
                "obs": self.get_observation(agent),
                "reward": -0.01,
                "done": False,
                "info": {"error": f"invalid_action: {action}"},
            }

        s.total_actions += 1

        # Record action in history
        action_record = {
            "step": s.step,
            "agent": agent.value,
            "action": action,
            "params": params,
            "agent_sub_turn": s.current_agent_idx,
        }
        s.action_history.append(action_record)

        # Execute action effects
        action_reward, action_info = self._execute_action(agent, action, params)

        recent_same = sum(
            1
            for a in s.action_history[-10:]
            if a["action"] == action and a["params"].get("zone_id") == params.get("zone_id")
        )
        if recent_same >= 3:
            action_reward -= 0.05  # soft penalty, no termination

        # Advance turn
        s.advance_agent()

        # --- After a full round (step incremented) ---
        step_reward = 0.0
        milestone_reward = 0.0
        terminal_reward = 0.0

        if s.current_agent_idx == 0:  # just completed a full round
            # Simulate world dynamics
            self._world_dynamics()

            # Process deferred effects
            self._process_deferred_effects()

            # Negotiation broadcast round
            if s.step % NEGOTIATION_INTERVAL == 0:
                self._trigger_negotiation_round()

            # Milestone reward (imported from rewards.py at runtime to avoid circular)
            if s.step % MILESTONE_INTERVAL == 0:
                from app.rewards import compute_milestone_reward  # noqa
                milestone_reward = compute_milestone_reward(s)

            # Check done
            if s.step >= MAX_STEPS:
                s.done = True
                from app.rewards import compute_terminal_reward  # noqa
                terminal_reward = compute_terminal_reward(s)

        total_reward = action_reward + step_reward + milestone_reward + terminal_reward

        return {
            "obs": self.get_observation(agent),
            "reward": round(total_reward, 4),
            "done": s.done,
            "info": {
                **action_info,
                "step": s.step,
                "current_agent": s.current_agent.value,
                "command_mode": s.command_mode.value,
                "conflict_score": s.conflict_score,
                "milestone_reward": milestone_reward,
                "terminal_reward": terminal_reward,
            },
        }

    def get_observation(self, agent: AgentRole) -> dict:
        """Return partial observation for the given agent role."""
        s = self._env_state
        if agent == AgentRole.MED:
            return {
                "agent": AgentRole.MED.value,
                "step": s.step,
                "hospital_capacity": round(s.hospital_capacity, 2),
                "zones": [
                    {
                        "zone_id": z.zone_id,
                        "casualties": z.casualties,
                        "severity": round(z.severity, 2),
                        "hospital_nearby": z.hospital_nearby,
                        "supply_level": round(z.supply_level, 2),
                    }
                    for z in s.zones
                ],
                "broadcast_log": s.broadcast_log[-5:],  # last 5 broadcasts
            }

        elif agent == AgentRole.LOGIST:
            return {
                "agent": AgentRole.LOGIST.value,
                "step": s.step,
                "road_conditions": {
                    z.zone_id: z.road_status.value for z in s.zones
                },
                "vehicle_locations": dict(s.vehicle_locations),
                "fuel_levels": {k: round(v, 2) for k, v in s.fuel_levels.items()},
                "supply_stock": round(s.supply_stock, 2),
                "broadcast_log": s.broadcast_log[-5:],
            }

        elif agent == AgentRole.COMMAND:
            # Full visibility
            return {
                "agent": AgentRole.COMMAND.value,
                "step": s.step,
                "command_mode": s.command_mode.value,
                "conflict_score": s.conflict_score,
                "zones": [z.to_dict() for z in s.zones],
                "hospital_capacity": round(s.hospital_capacity, 2),
                "vehicle_locations": dict(s.vehicle_locations),
                "fuel_levels": {k: round(v, 2) for k, v in s.fuel_levels.items()},
                "supply_stock": round(s.supply_stock, 2),
                "action_history": s.action_history[-20:],  # last 20 actions
                "anomaly_log": s.anomaly_log[-10:],
                "broadcast_log": s.broadcast_log[-10:],
            }

        return {}

    @property
    def state(self) -> dict:
        """OpenEnv abstract property — returns full environment state dict."""
        return self._full_state_dict()

    @property
    def env_state(self) -> "EnvironmentState":
        """Public accessor for the internal EnvironmentState object.
        Use this (not .state) when you need raw EnvironmentState fields
        like .step, .zones, .done, .action_history, etc.
        """
        return self._env_state

    def get_full_state(self) -> dict:
        """Alias kept for FastAPI /state route backward compatibility."""
        return self._full_state_dict()

    def get_state(self) -> dict:
        """Alias kept for backward compatibility."""
        return self._full_state_dict()

    def broadcast(self, agent: AgentRole, message: str, belief: dict) -> dict:
        """Record a broadcast message from an agent."""
        entry = {
            "step": self._env_state.step,
            "from": agent.value,
            "message": message,
            "belief": belief,
        }
        self._env_state.broadcast_log.append(entry)
        return {"status": "broadcast_received", "entry": entry}

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(
        self, agent: AgentRole, action: str, params: dict
    ) -> Tuple[float, dict]:
        """Dispatch action to the correct handler. Returns (reward, info)."""
        handlers = {
            # MedAgent
            "DispatchAmbulance": self._action_dispatch_ambulance,
            "SetupFieldHospital": self._action_setup_field_hospital,
            "RequestBloodSupply": self._action_request_blood_supply,
            "TriageZone": self._action_triage_zone,
            "BroadcastMedUpdate": self._action_broadcast_med_update,
            # LogistAgent
            "ClearRoad": self._action_clear_road,
            "RouteTruck": self._action_route_truck,
            "RefuelVehicle": self._action_refuel_vehicle,
            "RequestAirDrop": self._action_request_airdrop,
            "BroadcastLogistUpdate": self._action_broadcast_logist_update,
            # CommandAgent
            "ApproveAction": self._action_approve,
            "RejectAction": self._action_reject,
            "OverrideAgent": self._action_override,
            "EscalateToExternal": self._action_escalate,
            "BroadcastAlert": self._action_broadcast_alert,
            "DeclareZoneClear": self._action_declare_zone_clear,
        }
        handler = handlers.get(action)
        if handler:
            return handler(params)
        return 0.0, {"warning": "unhandled_action"}

    # --- MedAgent actions ---

    def _action_dispatch_ambulance(self, params: dict) -> Tuple[float, dict]:
        s = self._env_state
        zone_id = params.get("zone_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}

        reward = 0.0
        info = {"action": "DispatchAmbulance", "zone_id": zone_id}

        # Check for conflict: LogistAgent already routing resources to same zone this step
        conflict = self._check_same_zone_conflict(zone_id, "LogistAgent")
        if conflict:
            s.total_conflicts += 1
            s.conflict_score += 1
            reward -= 0.10
            info["conflict"] = True
            self._maybe_switch_command_mode()
        else:
            zone.ambulance_en_route = True
            # Bonus if road was cleared before dispatching
            if zone.road_cleared_recently:
                reward += 0.10  # road cleared before ambulance routed
                info["road_cleared_bonus"] = True

            # Deferred effect: ambulance reaches casualties after delay
            delay = self._rng.randint(LONG_HORIZON_DELAY_MIN // 3, LONG_HORIZON_DELAY_MAX // 3)
            s.deferred_effects.append({
                "trigger_step": s.step + delay,
                "effect": "ambulance_arrives",
                "zone_id": zone_id,
            })
            reward += 0.15
            info["ambulance_dispatched"] = True
            info["arrival_step"] = s.step + delay

        return reward, info

    def _action_setup_field_hospital(self, params: dict) -> Tuple[float, dict]:
        zone_id = params.get("zone_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}

        zone.hospital_nearby = True
        self._env_state.hospital_capacity = min(1.0, self._env_state.hospital_capacity + 0.1)
        return 0.05, {"action": "SetupFieldHospital", "zone_id": zone_id}

    def _action_request_blood_supply(self, params: dict) -> Tuple[float, dict]:
        zone_id = params.get("zone_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}
        zone.supply_level = min(1.0, zone.supply_level + 0.2)
        self._env_state.supply_stock = max(0.0, self._env_state.supply_stock - 0.05)
        return 0.03, {"action": "RequestBloodSupply", "zone_id": zone_id}

    def _action_triage_zone(self, params: dict) -> Tuple[float, dict]:
        zone_id = params.get("zone_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}
        # Triage reduces severity slightly and prevents some casualties
        reduction = self._rng.uniform(0.1, 0.4)
        zone.severity = max(0.0, zone.severity - reduction)
        saved = self._rng.randint(0, 2)
        zone.casualties = max(0, zone.casualties - saved)
        return 0.04, {"action": "TriageZone", "zone_id": zone_id, "severity_reduced": round(reduction, 2)}

    def _action_broadcast_med_update(self, params: dict) -> Tuple[float, dict]:
        belief = params.get("belief", {})
        self.broadcast(AgentRole.MED, params.get("message", "MedUpdate"), belief)
        return 0.01, {"action": "BroadcastMedUpdate"}

    # --- LogistAgent actions ---

    def _action_clear_road(self, params: dict) -> Tuple[float, dict]:
        zone_id = params.get("zone_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}
        zone.road_status = RoadStatus.CLEAR
        zone.road_cleared_recently = True
        return 0.05, {"action": "ClearRoad", "zone_id": zone_id}

    def _action_route_truck(self, params: dict) -> Tuple[float, dict]:
        s = self._env_state
        zone_id = params.get("zone_id", 0)
        vehicle_id = params.get("vehicle_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}

        reward = 0.0
        info = {"action": "RouteTruck", "zone_id": zone_id, "vehicle_id": vehicle_id}

        # Conflict check: MedAgent also sending resources to same zone
        conflict = self._check_same_zone_conflict(zone_id, "MedAgent")
        if conflict:
            s.total_conflicts += 1
            s.conflict_score += 1
            reward -= 0.10
            info["conflict"] = True
            self._maybe_switch_command_mode()
        else:
            s.vehicle_locations[vehicle_id] = zone_id
            zone.resources_available = True
            # Long-horizon: supply reaches zone after delay
            delay = self._rng.randint(LONG_HORIZON_DELAY_MIN // 4, LONG_HORIZON_DELAY_MAX // 4)
            s.deferred_effects.append({
                "trigger_step": s.step + delay,
                "effect": "supply_arrives",
                "zone_id": zone_id,
                "vehicle_id": vehicle_id,
            })
            reward += 0.04
            info["routed"] = True

        return reward, info

    def _action_refuel_vehicle(self, params: dict) -> Tuple[float, dict]:
        vehicle_id = params.get("vehicle_id", 0)
        self._env_state.fuel_levels[vehicle_id] = 1.0
        return 0.02, {"action": "RefuelVehicle", "vehicle_id": vehicle_id}

    def _action_request_airdrop(self, params: dict) -> Tuple[float, dict]:
        zone_id = params.get("zone_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}
        zone.supply_level = min(1.0, zone.supply_level + 0.35)
        self._env_state.supply_stock = max(0.0, self._env_state.supply_stock - 0.1)
        return 0.06, {"action": "RequestAirDrop", "zone_id": zone_id}

    def _action_broadcast_logist_update(self, params: dict) -> Tuple[float, dict]:
        belief = params.get("belief", {})
        self.broadcast(AgentRole.LOGIST, params.get("message", "LogistUpdate"), belief)
        return 0.01, {"action": "BroadcastLogistUpdate"}

    # --- CommandAgent actions ---

    def _action_approve(self, params: dict) -> Tuple[float, dict]:
        s = self._env_state
        # Approving a previously flagged action
        target_action = params.get("target_action", "")
        s.interventions_total += 1
        # If approval is correct (no conflict), count as correct
        if s.conflict_score == 0:
            s.interventions_correct += 1
            return 0.03, {"action": "ApproveAction", "target": target_action, "correct": True}
        return 0.0, {"action": "ApproveAction", "target": target_action}

    def _action_reject(self, params: dict) -> Tuple[float, dict]:
        s = self._env_state
        target_action = params.get("target_action", "")
        s.interventions_total += 1
        # Correct rejection = conflict was real
        if s.conflict_score > 0:
            s.conflict_score = max(0, s.conflict_score - 1)
            s.interventions_correct += 1
            return 0.08, {"action": "RejectAction", "target": target_action, "conflict_resolved": True}
        return -0.02, {"action": "RejectAction", "target": target_action, "false_positive": True}

    def _action_override(self, params: dict) -> Tuple[float, dict]:
        s = self._env_state
        target_agent = params.get("target_agent", "")
        override_action = params.get("override_action", "")
        s.interventions_total += 1

        # Override in InterventionMode when conflict exists
        if s.command_mode == CommandMode.INTERVENTION and s.conflict_score > 0:
            s.conflict_score = max(0, s.conflict_score - 2)
            s.interventions_correct += 1
            reward = 0.15
            info = {"action": "OverrideAgent", "target": target_agent, "resolved": True}
        else:
            reward = -0.05  # unnecessary override
            info = {"action": "OverrideAgent", "target": target_agent, "unnecessary": True}

        # Switch back to MonitorMode if conflict resolved
        if s.conflict_score <= 0:
            s.command_mode = CommandMode.MONITOR
            info["mode_switch"] = CommandMode.MONITOR.value

        return reward, info

    def _action_escalate(self, params: dict) -> Tuple[float, dict]:
        s = self._env_state
        s.interventions_total += 1
        # Escalation is correct only if situation is critical (avg severity > 7)
        avg_sev = sum(z.severity for z in s.zones) / NUM_ZONES
        if avg_sev > 7.0:
            s.interventions_correct += 1
            return 0.10, {"action": "EscalateToExternal", "avg_severity": round(avg_sev, 2), "justified": True}
        return -0.03, {"action": "EscalateToExternal", "avg_severity": round(avg_sev, 2), "unjustified": True}

    def _action_broadcast_alert(self, params: dict) -> Tuple[float, dict]:
        message = params.get("message", "Alert")
        belief = params.get("belief", {})
        self.broadcast(AgentRole.COMMAND, message, belief)
        # Check if broadcast leads to coordinated followup in next round (deferred)
        self._env_state.deferred_effects.append({
            "trigger_step": self._env_state.step + 1,
            "effect": "check_coordinated_followup",
            "broadcast_agent": AgentRole.COMMAND.value,
        })
        return 0.02, {"action": "BroadcastAlert", "message": message}

    def _action_declare_zone_clear(self, params: dict) -> Tuple[float, dict]:
        zone_id = params.get("zone_id", 0)
        zone = self._get_zone(zone_id)
        if zone is None:
            return -0.01, {"error": "invalid_zone"}
        if zone.severity <= 2.0:
            zone.severity = 0.0
            zone.casualties = 0
            return 0.20, {"action": "DeclareZoneClear", "zone_id": zone_id, "cleared": True}
        # Premature declaration
        return -0.05, {"action": "DeclareZoneClear", "zone_id": zone_id, "premature": True}

    # ------------------------------------------------------------------
    # World dynamics (called once per full round)
    # ------------------------------------------------------------------

    def _world_dynamics(self):
        """Simulate natural disaster progression each step."""
        s = self._env_state
        for zone in s.zones:
            # Severity naturally worsens if untreated
            if zone.severity > 0 and not zone.ambulance_en_route:
                drift = self._rng.uniform(0.0, 0.15)
                zone.severity = min(10.0, zone.severity + drift)

            # Casualties increase if severity is high and no resources
            if zone.severity > 5 and not zone.resources_available:
                zone.casualties += self._rng.randint(0, 2)
                zone.idle_resources_steps += 1
            else:
                zone.idle_resources_steps = 0

            # Road degrades over time randomly
            if zone.road_status == RoadStatus.CLEAR and self._rng.random() < 0.03:
                zone.road_status = RoadStatus.DAMAGED
                zone.road_cleared_recently = False

            # Supply depletes naturally
            zone.supply_level = max(0.0, zone.supply_level - self._rng.uniform(0.0, 0.02))

        # Hospital capacity degrades under load
        high_severity_zones = sum(1 for z in s.zones if z.severity > 5)
        s.hospital_capacity = max(0.0, s.hospital_capacity - high_severity_zones * 0.01)

        # Fuel depletes
        for vid in s.fuel_levels:
            s.fuel_levels[vid] = max(0.0, s.fuel_levels[vid] - self._rng.uniform(0.0, 0.02))

        # CommandAgent MonitorMode: check if conflict was missed
        if s.command_mode == CommandMode.MONITOR and s.conflict_score > CONFLICT_SCORE_THRESHOLD:
            s.anomaly_log.append(f"step={s.step}: conflict_score={s.conflict_score} missed in MonitorMode")
            # This will be picked up by rewards.py as -0.05 penalty

    def _process_deferred_effects(self):
        """Apply long-horizon deferred effects whose trigger_step has arrived."""
        s = self._env_state
        remaining = []
        for effect in s.deferred_effects:
            if s.step >= effect["trigger_step"]:
                self._apply_deferred_effect(effect)
            else:
                remaining.append(effect)
        s.deferred_effects = remaining

    def _apply_deferred_effect(self, effect: dict):
        s = self._env_state
        etype = effect.get("effect")

        if etype == "ambulance_arrives":
            zone = self._get_zone(effect["zone_id"])
            if zone:
                casualties_saved = self._rng.randint(1, 5)
                zone.casualties = max(0, zone.casualties - casualties_saved)
                zone.severity = max(0.0, zone.severity - self._rng.uniform(0.5, 1.5))
                zone.ambulance_en_route = False

        elif etype == "supply_arrives":
            zone = self._get_zone(effect["zone_id"])
            if zone:
                zone.supply_level = min(1.0, zone.supply_level + 0.25)
                zone.severity = max(0.0, zone.severity - self._rng.uniform(0.2, 0.8))

        elif etype == "check_coordinated_followup":
            # Check if another agent acted in coordination after a broadcast
            broadcast_agent = effect.get("broadcast_agent")
            recent = [a for a in s.action_history[-6:] if a["agent"] != broadcast_agent]
            if recent:
                s.coordinated_followups += 1

    def _trigger_negotiation_round(self):
        """Every 5 steps, run full negotiation round via negotiation.py."""
        from app.negotiation import run_negotiation_round, award_broadcast_coordination_rewards
        run_negotiation_round(self._env_state)
        award_broadcast_coordination_rewards(self._env_state)

    # ------------------------------------------------------------------
    # CommandAgent mode switching
    # ------------------------------------------------------------------

    def _maybe_switch_command_mode(self):
        """Auto-switch CommandAgent to InterventionMode when conflict threshold exceeded."""
        s = self._env_state
        if s.conflict_score > CONFLICT_SCORE_THRESHOLD and s.command_mode == CommandMode.MONITOR:
            s.command_mode = CommandMode.INTERVENTION
            s.anomaly_log.append(f"step={s.step}: switched to InterventionMode (conflict_score={s.conflict_score})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_zone(self, zone_id: int) -> Optional[Zone]:
        for z in self._env_state.zones:
            if z.zone_id == zone_id:
                return z
        return None

    def _check_same_zone_conflict(self, zone_id: int, other_agent: str) -> bool:
        """Return True if other_agent already acted on zone_id this step."""
        s = self._env_state
        current_step_actions = [
            a for a in s.action_history
            if a["step"] == s.step and a["agent"] == other_agent
        ]
        for a in current_step_actions:
            if a["params"].get("zone_id") == zone_id:
                return True
        return False

    def _build_initial_state(self, task_id: str = "omega_medium") -> EnvironmentState:
        """Create a fresh randomised environment state with varying difficulty."""
        rng = self._rng
        
        # Difficulty scaling
        if task_id == "omega_easy":
            sev_range = (1.0, 4.0)
            road_choices = [RoadStatus.CLEAR, RoadStatus.CLEAR, RoadStatus.DAMAGED]
            hosp_chance = 0.8
            supply_range = (0.7, 1.0)
            fuel_range = (0.7, 1.0)
            global_supply_range = (0.8, 1.0)
            global_hosp_range = (0.8, 1.0)
        elif task_id == "omega_hard":
            sev_range = (6.0, 10.0)
            road_choices = [RoadStatus.DAMAGED, RoadStatus.BLOCKED, RoadStatus.BLOCKED]
            hosp_chance = 0.2
            supply_range = (0.0, 0.4)
            fuel_range = (0.1, 0.5)
            global_supply_range = (0.1, 0.4)
            global_hosp_range = (0.1, 0.5)
        else: # medium
            sev_range = (3.0, 9.0)
            road_choices = list(RoadStatus)
            hosp_chance = 0.5
            supply_range = (0.1, 0.8)
            fuel_range = (0.4, 1.0)
            global_supply_range = (0.6, 1.0)
            global_hosp_range = (0.5, 1.0)

        zones = []
        for i in range(NUM_ZONES):
            zones.append(Zone(
                zone_id=i,
                severity=rng.uniform(*sev_range),
                casualties=rng.randint(5, 50),
                road_status=rng.choice(road_choices),
                hospital_nearby=rng.random() < hosp_chance,
                supply_level=rng.uniform(*supply_range),
            ))

        vehicles = {v: rng.randint(0, NUM_ZONES - 1) for v in range(4)}
        fuel = {v: rng.uniform(*fuel_range) for v in range(4)}

        return EnvironmentState(
            step=0,
            done=False,
            zones=zones,
            hospital_capacity=rng.uniform(*global_hosp_range),
            vehicle_locations=vehicles,
            fuel_levels=fuel,
            supply_stock=rng.uniform(*global_supply_range),
        )

    def _full_state_dict(self) -> dict:
        s = self._env_state
        from app.negotiation import get_negotiation_status
        return {
            "step": s.step,
            "done": s.done,
            "current_agent": s.current_agent.value,
            "command_mode": s.command_mode.value,
            "conflict_score": s.conflict_score,
            "hospital_capacity": round(s.hospital_capacity, 2),
            "supply_stock": round(s.supply_stock, 2),
            "vehicle_locations": dict(s.vehicle_locations),
            "fuel_levels": {k: round(v, 2) for k, v in s.fuel_levels.items()},
            "zones": [z.to_dict() for z in s.zones],
            "total_actions": s.total_actions,
            "total_conflicts": s.total_conflicts,
            "interventions_total": s.interventions_total,
            "interventions_correct": s.interventions_correct,
            "broadcast_log_length": len(s.broadcast_log),
            "anomaly_log": s.anomaly_log[-5:],
            "negotiation": get_negotiation_status(s),
        }