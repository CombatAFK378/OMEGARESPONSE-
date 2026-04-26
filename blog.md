# OmegaResponse: What We Actually Built, Broke, and Fixed in Bangalore

Built for the Meta PyTorch OpenEnv Grand Finale, Bangalore 2026.

This is not a polished lab write-up. This is what happened when we tried to make three AI agents coordinate in a disaster simulation while the clock was running.

## Why This Problem Felt Real
In disaster response, mistakes are expensive. If medical and logistics teams optimize for their own local view, they can still fail globally. One team can do the right thing in isolation and still create gridlock for everyone else.

That was the core pain point we wanted OmegaResponse to model.

OmegaResponse is a multi-agent environment with three roles:
1. MedAgent for casualties and field care decisions.
2. LogistAgent for roads, routing, and supplies.
3. CommandAgent for conflict detection and intervention.

Each agent has partial observability. No single agent sees the full world at once. They must share intent through broadcasts, and the consequences of one action can appear many steps later. That long delay is exactly what makes this hard.

## What We Ran (Correct Setup)
We trained in JupyterLab on an A100 GPU, not in a lightweight smoke setup.

Training details:
1. Model family: Qwen 2.5 instruction-tuned base.
2. Optimizer and GRPO setup via TRL.
3. Reward shaping from app/rewards.py with strict parsing and format checks.
4. Total training length: 600 steps.

The 600-step run mattered. Short runs looked fine in a few examples but did not hold under harder scenarios. At 600, behavior became noticeably more stable.

## The Most Important Engineering Decision
The biggest lift was reward design.

If rewards were too soft, the model learned to produce pretty outputs without operational quality. If rewards were too harsh, learning became unstable.

We pushed toward stricter signals in app/rewards.py:
1. Action validity with stronger positive and negative feedback.
2. Format compliance that distinguishes full structured outputs from partial ones.
3. Zone-priority bonuses for targeting the right zone under pressure.
4. Primary heuristic reward that aligns with actual disaster response goals.

This was the difference between "valid-looking text" and "useful action policy."

## What Changed in Agent Behavior
Before tuning, policies were brittle and often repetitive. In hard conditions, the system drifted toward safe but low-value actions.

After tuning with the stricter reward stack:
1. Outputs followed the expected structured action format more reliably.
2. Agents targeted higher-severity zones more consistently.
3. CommandAgent intervened more appropriately when conflict_score crossed threshold.
4. Cross-agent coordination improved under high-stress states.

Not perfect, but meaningfully better.

## OOD Reality Check
We specifically tested unseen, worst-case prompts: high severity across zones, blocked roads, low hospital capacity, and conflict-heavy action histories.

Rule-based logic stayed useful as a baseline, but it plateaued quickly when conditions combined in ways it did not explicitly encode. The fine-tuned policy handled those combinations better because it learned a smoother decision boundary from many state variants.

That was our main success criterion: not just fitting seen prompts, but behaving sanely when the scenario got ugly.

## What This Project Means to Us
This project stopped feeling like a hackathon toy the moment we watched one bad routing choice cascade into avoidable failures across the simulated city.

The emotional part was simple: in real crises, coordination failures are human failures.

OmegaResponse is our attempt to push toward AI that does not just answer correctly, but coordinates responsibly under uncertainty.

## Final Takeaway
With JupyterLab on A100, a 600-step GRPO run, and stricter reward engineering, OmegaResponse moved from a fragile demo to a more credible coordination policy.

If there is one lesson we would keep, it is this: in multi-agent disaster systems, reward design is not tuning polish. It is the system.
