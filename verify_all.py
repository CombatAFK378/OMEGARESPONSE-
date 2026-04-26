#!/usr/bin/env python3
"""
OmegaResponse — Pre-Bangalore Verification Script
Run from project root: python verify_all.py
Checks all imports, cross-file wiring, and basic functionality.
"""

import sys, traceback

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  ❌ {name}")
        print(f"     {e}")

print("=" * 55)
print("  OmegaResponse Pre-Bangalore Verification")
print("=" * 55)

# ── 1. Core imports ──────────────────────────────────────
print("\n[1] Import checks")

def test_rewards_import():
    from app.rewards import (
        compute_grpo_reward, reward_action_validity,
        reward_format_compliance, reward_zone_priority
    )

def test_environment_import():
    from app.environment import DisasterEnvironment, AgentRole

def test_negotiation_import():
    from app.negotiation import (
        run_negotiation_round, get_negotiation_status,
        award_broadcast_coordination_rewards
    )

def test_agents_import():
    from app.agents import parse_agent_response, get_agent_action, AGENT_CONFIG

def test_main_import():
    import app.main  # checks FastAPI app boots without error

check("rewards.py imports", test_rewards_import)
check("environment.py imports", test_environment_import)
check("negotiation.py imports", test_negotiation_import)
check("agents.py imports", test_agents_import)
check("main.py imports (FastAPI)", test_main_import)

# ── 2. Environment lifecycle ─────────────────────────────
print("\n[2] Environment lifecycle")

def test_env_reset():
    from app.environment import DisasterEnvironment
    env = DisasterEnvironment()
    state = env.reset(seed=42)
    assert "step" in state
    assert "zones" in state
    assert len(state["zones"]) == 5

def test_env_step():
    from app.environment import DisasterEnvironment, AgentRole
    env = DisasterEnvironment()
    env.reset(seed=42)
    result = env.step(AgentRole.MED, "DispatchAmbulance", {"zone_id": 0})
    assert "obs" in result
    assert "reward" in result
    assert "done" in result
    assert isinstance(result["reward"], float)

def test_env_state_method():
    from app.environment import DisasterEnvironment
    env = DisasterEnvironment()
    env.reset(seed=42)
    # Check both get_full_state and state() exist
    s1 = env.get_full_state()
    assert "step" in s1
    s2 = env.get_state()  # OpenEnv required method
    assert "step" in s2

def test_env_full_rotation():
    from app.environment import DisasterEnvironment, AgentRole
    env = DisasterEnvironment()
    env.reset(seed=42)
    # Full round: Med -> Logist -> Command
    env.step(AgentRole.MED,     "DispatchAmbulance",  {"zone_id": 1})
    env.step(AgentRole.LOGIST,  "ClearRoad",          {"zone_id": 2})
    result = env.step(AgentRole.COMMAND, "BroadcastAlert",  {"message": "test", "belief": {}})
    assert result["info"]["step"] >= 1

check("env.reset()", test_env_reset)
check("env.step()", test_env_step)
check("env.state() method exists", test_env_state_method)
check("full Med→Logist→Command rotation", test_env_full_rotation)

# ── 3. Negotiation wiring ────────────────────────────────
print("\n[3] Negotiation wiring")

def test_negotiation_called():
    from app.environment import DisasterEnvironment, AgentRole
    env = DisasterEnvironment()
    env.reset(seed=42)
    # Run 5 steps to trigger negotiation round
    agents = [AgentRole.MED, AgentRole.LOGIST, AgentRole.COMMAND]
    for i in range(15):  # 5 full rounds
        agent = agents[i % 3]
        action = ["DispatchAmbulance","ClearRoad","BroadcastAlert"][i % 3]
        params = {"zone_id": 0, "message": "test", "belief": {}}
        env.step(agent, action, params)
    # After 5 steps, negotiation log should have entries
    neg_entries = [b for b in env.env_state.broadcast_log if b.get("type") == "negotiation_round"]
    assert len(neg_entries) > 0, "negotiation_round never triggered"

check("negotiation triggers after 5 steps", test_negotiation_called)

# ── 4. Reward functions ──────────────────────────────────
print("\n[4] Reward functions")

def test_compute_grpo_reward():
    from app.rewards import compute_grpo_reward
    completions = ["Action: DispatchAmbulance\nParameters: {\"zone_id\": 1}\nReasoning: highest severity"]
    prompts = ["You are the MedAgent. Zone 1: severity=8.5 | casualties=30 | hospital_nearby=False | supply_level=0.2"]
    rewards = compute_grpo_reward(completions, prompts=prompts)
    assert len(rewards) == 1
    assert isinstance(rewards[0], float)
    assert rewards[0] > 0, f"Expected positive reward, got {rewards[0]}"

def test_reward_heads():
    from app.rewards import reward_action_validity, reward_format_compliance, reward_zone_priority
    c = ["Action: ClearRoad\nParameters: {\"zone_id\": 2}"]
    p = ["Zone 0: severity=3.0 Zone 2: severity=8.5"]
    assert reward_action_validity(c)[0] > 0
    assert reward_format_compliance(c)[0] > 0
    assert reward_zone_priority(c, prompts=p)[0] > 0

def test_milestone_reward():
    from app.environment import DisasterEnvironment, AgentRole
    from app.rewards import compute_milestone_reward
    env = DisasterEnvironment()
    env.reset(seed=42)
    r = compute_milestone_reward(env.env_state)
    assert isinstance(r, float)

def test_terminal_reward():
    from app.rewards import compute_terminal_reward
    from app.environment import DisasterEnvironment
    env = DisasterEnvironment()
    env.reset(seed=42)
    r = compute_terminal_reward(env.env_state)
    assert isinstance(r, float)

check("compute_grpo_reward()", test_compute_grpo_reward)
check("reward_action_validity / format / zone_priority", test_reward_heads)
check("compute_milestone_reward()", test_milestone_reward)
check("compute_terminal_reward()", test_terminal_reward)

# ── 5. Agents ────────────────────────────────────────────
print("\n[5] Agent prompt builders & parser")

def test_agent_prompts():
    from app.agents import build_med_prompt, build_logist_prompt, build_command_prompt
    from app.environment import DisasterEnvironment, AgentRole
    env = DisasterEnvironment()
    env.reset(seed=42)
    med_obs   = env.get_observation(AgentRole.MED)
    logist_obs = env.get_observation(AgentRole.LOGIST)
    cmd_obs   = env.get_observation(AgentRole.COMMAND)
    assert "Zone" in build_med_prompt(med_obs)
    assert "ROAD" in build_logist_prompt(logist_obs)
    assert "ZONE STATUS" in build_command_prompt(cmd_obs)

def test_parse_agent_response():
    from app.agents import parse_agent_response
    raw = "Action: OverrideAgent\nParameters: {\"target_agent\": \"MedAgent\", \"zone_id\": 3}\nReasoning: conflict"
    result = parse_agent_response(raw)
    assert result["action"] == "OverrideAgent"
    assert result["params"].get("zone_id") == 3

def test_rule_based_action():
    from app.agents import get_agent_action
    from app.environment import DisasterEnvironment, AgentRole
    env = DisasterEnvironment()
    env.reset(seed=42)
    obs = env.get_observation(AgentRole.MED)
    result = get_agent_action("MedAgent", obs, use_groq=False)
    assert result["action"] != ""
    assert "zone_id" in result["params"]

check("prompt builders (Med/Logist/Command)", test_agent_prompts)
check("parse_agent_response()", test_parse_agent_response)
check("rule-based get_agent_action()", test_rule_based_action)

# ── 6. main.py (DisasterEnvironment() no-seed init) ──────
print("\n[6] main.py init check")

def test_main_env_init():
    # Verify ENV in main.py was created without seed arg
    import app.main as m
    assert hasattr(m, "ENV")
    assert m.ENV is not None
    assert m.ENV.env_state.step == 0

check("main.py ENV initialises correctly", test_main_env_init)

# ── Summary ──────────────────────────────────────────────
print()
print("=" * 55)
total = len(PASS) + len(FAIL)
print(f"  Results: {len(PASS)}/{total} passed")
print("=" * 55)

if FAIL:
    print("\n  FAILED CHECKS:")
    for name, err in FAIL:
        print(f"  ❌ {name}")
        print(f"     Fix: {err[:120]}")
    print()
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED — safe to push and sleep 🚀")
    sys.exit(0)
