# episode_runner.py
# Runs a full OmegaResponse episode using Groq LLM as the agent.
# Used ONLY for local testing before the Bangalore hackathon.
# Zero Groq calls are made during GRPO training.
#
# Prerequisites:
#   export GROQ_API_KEY=gsk_...
#   pip install groq
#
# Usage:
#   python episode_runner.py                        # full 300-step episode
#   python episode_runner.py --steps 30             # short smoke-test
#   python episode_runner.py --compare              # rule-based vs LLM side-by-side

import argparse
import os
import time
from pathlib import Path

from app.environment import DisasterEnvironment, AgentRole
from app.agents import (
    AGENT_CONFIG,
    parse_agent_response,
    call_llm,
    get_agent_action,
)
from rule_based_agent import run_rule_based_episode  # noqa — file written at step 6


# ---------------------------------------------------------------------------
# LLM episode runner
# ---------------------------------------------------------------------------

def run_llm_episode(
    seed: int = 42,
    max_steps: int = 300,
    verbose: bool = True,
    model: str = "llama-3.1-8b-instant",
) -> float:
    """
    Run one full episode where each agent is driven by Groq LLM.
    Returns normalized per-action score.

    Agent turn order per step: MedAgent → LogistAgent → CommandAgent
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Run: $env:GROQ_API_KEY='gsk_...'  (PowerShell)\n"
            "  or: export GROQ_API_KEY=gsk_...  (bash)"
        )

    env = DisasterEnvironment(seed=seed)
    env.reset(seed=seed)

    total_reward = 0.0
    step_rewards = []
    rotation = [AgentRole.MED, AgentRole.LOGIST, AgentRole.COMMAND]
    role_names = {
        AgentRole.MED:     "MedAgent",
        AgentRole.LOGIST:  "LogistAgent",
        AgentRole.COMMAND: "CommandAgent",
    }

    done = False
    sub_turn = 0
    groq_calls = 0
    errors = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  OmegaResponse — LLM Episode (Groq / {model})")
        print(f"  seed={seed}  max_steps={max_steps}")
        print(f"{'='*60}\n")

    while not done and env.state.step < max_steps:
        agent = rotation[sub_turn % 3]
        role_name = role_names[agent]
        obs = env.get_observation(agent)

        # Build prompt
        config = AGENT_CONFIG[role_name]
        user_prompt = config["build_prompt"](obs)
        system_prompt = config["system_prompt"]

        # Call LLM
        try:
            raw_response = call_llm(system_prompt, user_prompt, model=model)
            groq_calls += 1
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Groq call failed at step {env.state.step}: {e}")
            # Fall back to rule-based action on API error
            parsed = get_agent_action(role_name, obs, use_groq=False)
            errors += 1
            raw_response = ""
        else:
            parsed = parse_agent_response(raw_response)

        action = parsed["action"]
        params = parsed["params"]

        # Execute in environment
        result = env.step(agent, action, params)
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward
        step_rewards.append(reward)

        # Verbose logging — every step so you can see progress
        if verbose and sub_turn % 3 == 2:
            info = result.get("info", {})
            print(
                f"  Step {env.state.step:3d}/300 | "
                f"{role_name:12s} | "
                f"action={action:22s} | "
                f"zone={params.get('zone_id', '-')} | "
                f"reward={reward:+.4f} | "
                f"cumulative={total_reward:.3f} | "
                f"mode={info.get('command_mode', '?')}"
            )
            if parsed.get("reasoning"):
                print(f"    Reasoning: {parsed['reasoning'][:80]}")

        sub_turn += 1
        # Small sleep to avoid Groq rate limits
        time.sleep(0.8)   # 800ms between calls — stays under 6000 TPM free tier

    if verbose:
        print(f"\n{'='*60}")
        print(f"  EPISODE COMPLETE")
        print(f"  Total reward      : {total_reward:.4f}")
        print(f"  Avg reward/step   : {total_reward / max(len(step_rewards), 1):.4f}")
        print(f"  Groq API calls    : {groq_calls}")
        print(f"  Fallback errors   : {errors}")
        print(f"  Zones recovered   : {sum(1 for z in env.state.zones if z.severity == 0.0)}/5")
        print(f"  Total conflicts   : {env.state.total_conflicts}")
        print(f"  Interventions     : {env.state.interventions_correct}/{env.state.interventions_total} correct")
        print(f"{'='*60}\n")

    normalized = total_reward / max(sub_turn, 1)
    if verbose:
        print(f"  Normalized per-action : {round(normalized, 4)}")
    return round(normalized, 4)


# ---------------------------------------------------------------------------
# Side-by-side comparison (for demo storytelling)
# ---------------------------------------------------------------------------

def run_comparison(seed: int = 42, max_steps: int = 60) -> None:
    """
    Run both rule-based and LLM episode on the same seed.
    Prints side-by-side score comparison — useful for demo prep.
    max_steps=60 to keep Groq costs low during testing.
    """
    print("\n" + "="*60)
    print("  COMPARISON: Rule-Based vs LLM Agent")
    print("="*60)

    print("\n[1/2] Running rule-based baseline...")
    rule_score = run_rule_based_episode(seed=seed, verbose=False)
    print(f"  Rule-based normalized score : {rule_score:.4f} per action")

    print(f"\n[2/2] Running LLM agent (Groq)...")
    llm_score = run_llm_episode(seed=seed, max_steps=max_steps, verbose=False)
    print(f"  LLM agent normalized score  : {llm_score:.4f} per action")

    print(f"\n{'='*60}")
    improvement = ((llm_score - rule_score) / abs(rule_score) * 100) if rule_score != 0 else 0
    print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"  Post-GRPO target: ~0.5-0.8 per action  (~3-4x baseline)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OmegaResponse LLM episode runner")
    parser.add_argument(
        "--mode",
        choices=["llm", "rule", "compare"],
        default="llm",
        help=(
            "llm     = run LLM (Groq) episode\n"
            "rule    = run rule-based baseline\n"
            "compare = run both and show score delta"
        ),
    )
    parser.add_argument("--seed",  type=int, default=42,  help="Random seed")
    parser.add_argument(
        "--steps",
        "--max-steps",
        dest="steps",
        type=int,
        default=300,
        help="Max steps (default 300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="Groq model name",
    )
    args = parser.parse_args()

    if args.mode == "llm":
        run_llm_episode(seed=args.seed, max_steps=args.steps, model=args.model)
    elif args.mode == "rule":
        score = run_rule_based_episode(seed=args.seed, verbose=True)
        print(f"Final score: {score:.4f}")
    elif args.mode == "compare":
        run_comparison(seed=args.seed, max_steps=args.steps)


if __name__ == "__main__":
    main()