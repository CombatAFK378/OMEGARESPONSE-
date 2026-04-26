---
title: OmegaResponse
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: true
---

# OmegaResponse 🚨

A 72-hour multi-agency disaster response simulation — an OpenEnv-compliant RL environment where three specialized LLM agents (MedAgent, LogistAgent, CommandAgent) coordinate rescue operations across 5 disaster zones over 300 steps with partial observability, agent negotiation, and sparse delayed rewards.

Built for the **Meta PyTorch OpenEnv Hackathon Grand Finale** — Bangalore, April 25–26 2026.

---

## 🔗 Official Deliverable Links

| Deliverable | Link |
|-------------|------|
| **Live Hugging Face Space** | [HuggingFace Space](https://huggingface.co/spaces/CombatAFK/OmegaResponse) |
| **GitHub Repository** | [GitHub - CombatAFK378/OMEGARESPONSE-](https://github.com/CombatAFK378/OMEGARESPONSE-) |
| **Colab Training Notebook** | [Open in Colab](https://colab.research.google.com/github/CombatAFK378/OMEGARESPONSE-/blob/main/training/grpo_train.ipynb) |
| **Video Walkthrough / Blog** | [Project Blog on HF](https://huggingface.co/blog/CombatAFK/omega-response) (or local [blog.md](blog.md)) |

*(Make sure to replace the placeholder links above before final submission!)*

---

## 📊 Training Results (Qwen 2.5 via GRPO)

We successfully fine-tuned Qwen 2.5 (3B) using the HF TRL GRPOTrainer. The model learned to strictly adhere to the OpenEnv JSON-style action formatting while simultaneously learning to prioritize zones based on severity and conflict avoidance.

### Standard Environment (Trained)
| Stage | Normalized score (per action) |
|-------|-------------------------------|
| Rule-based baseline | 0.1752 |
| Fine-tuned Qwen 2.5 | 0.2341 |
| **Improvement** | **+33%** |

### Out-Of-Distribution (OOD) Worst-Case Scenario
To prove the model generalizes and doesn't just memorize the state space, we evaluated it against an extreme worst-case disaster scenario (all zones critical, roads blocked, 21% hospital capacity).
| Stage | Normalized score (per action) |
|-------|-------------------------------|
| Rule-based agent (OOD) | 0.0657 |
| Fine-tuned Qwen 2.5 (OOD) | 0.0818 |
| **Verdict** | **GENERALIZES ✅ (+0.016)** |

### Visual Proof of Training

**Reward Curve:**
![Reward Curve](reward_curve.png)

**Loss Curve:**
![Loss Curve](loss_curve.png)

**OOD Generalization (Baseline vs Fine-tuned):**
![OOD Comparison](ood_comparison.png)

---

## 🌍 Live Environment

The `/state` endpoint shows the live disaster environment state:
- **5 disaster zones** — each with severity (0-10), casualties, road status, hospital proximity, supply level
- **3 agent roles** — MedAgent (WASH/WHO), LogistAgent (WFP Logistics), CommandAgent (OCHA Coordinator)
- **CommandAgent mode** — MonitorMode (passive) or InterventionMode (auto-triggers on conflict)
- **Negotiation rounds** — agents broadcast beliefs every 5 steps, conflicts detected automatically
- **Reward layers** — dense (every step) + milestone (every 20 steps) + terminal (step 300)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Container health check — returns `{"status": "ok"}` |
| GET | `/state` | Full environment state — zones, agents, scores, negotiation log |
| POST | `/reset` | Reset environment to a fresh randomized disaster scenario |
| POST | `/step` | Advance one step — body: `{"agent": "MedAgent", "action": "DispatchAmbulance", "parameters": {"zone_id": 3}}` |
| POST | `/broadcast` | Agent broadcasts belief — body: `{"from": "MedAgent", "message": "...", "belief": {}, "priority_zone": 3}` |

---

## 🏆 Hackathon Themes Covered

| Theme | Coverage |
|-------|----------|
| **Multi-Agent Interactions (40%)** | 3 independent agents with a shared global state, forced to negotiate via broadcasts. |
| **Long-Horizon Planning (30%)** | 300 steps. Actions like `DispatchAmbulance` take 40-100 steps to yield terminal rewards. |
| **Fleet AI (Bonus)** | CommandAgent monitors the other two agents passively and switches to InterventionMode to override conflicts. |
| **Scale AI (Bonus)** | Simulates complex non-coding real-world workflows (disaster supply chain logistics). |

---

## 🤖 Agents

### MedAgent — WASH Cluster (WHO)
- Sees: hospital capacity, casualties per zone, medical supplies
- Actions: `DispatchAmbulance`, `SetupFieldHospital`, `RequestBloodSupply`, `TriageZone`, `BroadcastMedUpdate`

### LogistAgent — Logistics Cluster (WFP)
- Sees: road conditions, vehicle locations, fuel levels, supply stock
- Actions: `ClearRoad`, `RouteTruck`, `RefuelVehicle`, `RequestAirDrop`, `BroadcastLogistUpdate`

### CommandAgent — OCHA Humanitarian Coordinator
- Sees: full state + action history of both agents
- Modes: MonitorMode (passive) -> auto-switches to InterventionMode when conflict score > 2
- Actions: `ApproveAction`, `RejectAction`, `OverrideAgent`, `EscalateToExternal`, `BroadcastAlert`, `DeclareZoneClear`

---

## 💻 Local Setup

```bash
git clone https://github.com/CombatAFK378/OMEGARESPONSE-.git
cd OMEGARESPONSE-
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
